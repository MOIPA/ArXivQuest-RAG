# Author：TangRui
# Date：2024/12/16
# 思路
# 
# 1. 构建预处理chain：两个chain，第一个chain翻译问题，第二个chain拓展优化问题，并说明问题所属领域
# 2. 根据优化后的问题，查找向量数据库，找出最合适的abstract
# 3. 构建一个routerchain将用户问题分类（6个领域）,选择最好的那个chain结合材料，回答问题
# 4. 将问题和回答让llm整理一遍，通过输出解析器得到格式化的问题和答案，将问题和答案输入llm返回一个true/false，确认是否偏题
# 5. 如果偏题就重新从1开始再来一遍
# 6. 最后如果没有问题，将信息转为中文并且封装信息

from langchain.chat_models import ChatOpenAI
import os
from langchain.embeddings import HuggingFaceEmbeddings
import json
from langchain.vectorstores import Milvus
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

class ArXivQA:
    def __init__(self,language='Chinese',embedding=None,llm_model = "Qwen2.5-14B",key="None",api_base="http://10.58.0.2:8000/v1",collection_name='arXiv'
                 ,milvus_host="10.58.0.2",port="19530"):
        self.language = language
        os.environ["OPENAI_API_KEY"] = key
        os.environ["OPENAI_API_BASE"] = api_base
        self.answer = None
        self.optimized_question = None
        self.original_question=None
        self.llm = ChatOpenAI(temperature=0, model=llm_model)
        if embedding == None:
            embedding = HuggingFaceEmbeddings(model_name="./all-MiniLM-L12-v2")
        self.db = Milvus(embedding_function=embedding,collection_name=collection_name,connection_args={"host": milvus_host, "port": port})
        
    def db_search(self,question,limits=3):
        return self.db.similarity_search(question,k=limits)
    
    def display(self,results,indent=4):
        dict_result = [doc.to_dict() if hasattr(doc, 'to_dict') else vars(doc) for doc in results]
        print(json.dumps(dict_result, indent=indent))

    def search_for_doc(self):
        doc = self.db_search(self.optimized_question,1)
        if doc==None or len(doc)<=0:
            return ""
        access_id = doc[0].metadata['access_id']
        authors = doc[0].metadata['authors']
        title = doc[0].metadata['title']
        source = f'authors:{authors}\ntitle:{title}\nwebsite:https://arxiv.org/abs/{access_id}\n'
        self.source=source
        self.abstract_doc = doc[0].page_content

    def choose_field_answer_based_doc(self):
        baseTemplate = """And must answer the questions based on a paper abstract provided to you.
        Questions:{input}
        Paper Abstract:
        """+self.abstract_doc

        NLP_template = """
        Please provide a detailed and comprehensive explanation of [question topic, e.g., large language models, their scaling laws, or instruction tuning]. Include its fundamental concepts, working mechanisms, typical applications, and any recent advancements or challenges in the field of natural language processing. If relevant, compare it with other similar techniques and discuss its future prospects.
        """+baseTemplate
        SE_template = """
        Describe [question topic, like formal software engineering, code review goals, or software engineering adaptation across fields] in-depth. Explain the key principles, methodologies, and best practices. Elaborate on how it fits into the overall software development life cycle, its importance for ensuring software quality and maintainability, and any industry standards or trends related to it.
        """+baseTemplate
        DL_template = """
        Analyze the impact of [question topic, such as duplicate data on In-content Learning] from multiple perspectives. Discuss the technical implications on learning algorithms, data processing pipelines, and model performance. Mention any strategies to mitigate negative effects and leverage positive aspects, if applicable, along with relevant case studies or research findings in the realm of data and machine learning
        """+baseTemplate
        BC_template= """
        Give a thorough account of how [question topic, e.g., blockchain ensures security] by detailing the underlying cryptographic techniques, consensus mechanisms, and network architectures involved. Explain the security threats it aims to counter and how it compares to traditional security models. Provide real-world examples of blockchain implementations with enhanced security features
        """+baseTemplate
        QC_template = """
        Elucidate the principle of [question topic, like ion-trap computers] in the context of quantum computing. Describe the quantum mechanical phenomena utilized, the hardware components and their functions, and the computational advantages it offers over classical computing. Discuss the current state-of-the-art research and potential future breakthroughs in this area.
        """+baseTemplate
        MSP_template = """
        Explain what [question topic, i.e., artificial atoms] are, covering their physical properties, synthetic methods, and potential applications. Compare them with natural atoms in terms of structure, behavior, and functionality. Cite relevant scientific literature or experimental results to support your explanation and discuss any emerging research directions
        """+baseTemplate
        prompt_infos = [
            {
                "name": "NLP", 
                "description": "Good for answering questions about Natural Language Processing and machine learning.", 
                "prompt_template": NLP_template
            },
            {
                "name": "SE", 
                "description": "Good for answering Software Engineering Field questions", 
                "prompt_template": SE_template
            },
            {
                "name": "DL", 
                "description": "Good for answering Data and Learning Field questions", 
                "prompt_template": DL_template
            },
            {
                "name": "BC", 
                "description": "Good for answering Blockchain Field questions", 
                "prompt_template": BC_template
            },
            {
                "name": "QC", 
                "description": "Good for answering Quantum Computing Field questions", 
                "prompt_template": QC_template
            },
            {
                "name": "MSP", 
                "description": "Good for answering Materials Science or Physics Field questions", 
                "prompt_template": MSP_template
            }
        ]

        # 准备所有子链给路由链决定
        destination_chains = {}
        for p_info in prompt_infos:
            # 读取出对应的prompt模板
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = ChatPromptTemplate.from_template(template=prompt_template)
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain  
            
        # 给定名字和子链的描述，让路由链决定哪个name更合适回答当前的问题
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        # 路由链实在找不到可用的子链，就用默认链
        default_prompt = ChatPromptTemplate.from_template("{input}")
        default_chain = LLMChain(llm=self.llm, prompt=default_prompt)


        # 路由链需要两个输入，所有子链和对应描述，当前任务

        MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input and relating document to a \
        language model select the model prompt best suited for the input. \
        You will be given the names of the available prompts and a \
        description of what the prompt is best suited for. \
        You may also revise the original input if you think that revising\
        it will ultimately lead to a better response from the language model.

        << FORMATTING >>
        Return a markdown code snippet with a JSON object formatted to look like:
        ```json
        {{{{
            "destination": string \ name of the prompt to use or "DEFAULT"
            "next_inputs": string \ a potentially modified version of the original input
        }}}}
        ```

        REMEMBER: "destination" MUST be one of the candidate prompt \
        names specified below OR it can be "DEFAULT" if the input is not\
        well suited for any of the candidate prompts.
        REMEMBER: "next_inputs" can just be the original input \
        if you don't think any modifications are needed.

        << CANDIDATE PROMPTS >>
        {destinations}

        << INPUT >>
        {{input}}

        << OUTPUT (remember to include the ```json)>>"""

        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str,
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )

        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)

        chain = MultiPromptChain(router_chain=router_chain, 
                                destination_chains=destination_chains, 
                                default_chain=default_chain,
                                verbose=True
                                )

        self.answer = chain.run(self.optimized_question)

    def format_result(self):
        # 定义回复的所有属性约束，就是对每个字段给定说明，和取值
        field_schema = ResponseSchema(name="question field",
                                    description="What field is this about? \
                                            choose from the following fields:\
                                            Natural Language Processing Field\
                                            or Software Engineering Field\
                                            or Data and Learning Field\
                                            or Blockchain Field\
                                            or Quantum Computing Field\
                                            or Materials Science or Physics Field")
        answer_schema = ResponseSchema(name="final answer",
                                            description="the final answer of the question.")
        question_schema = ResponseSchema(name="question",
                                            description="what is the quesiton of the text.")
        response_schemas = [question_schema,field_schema, answer_schema]

        # 定义输出解析器
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        # 获得格式化输出
        format_instructions = output_parser.get_format_instructions()
        # 使用输出解析器的提示词
        review_template_2 = """\
        For the following text, extract the following information:

        question field: What field is this about? \
                choose from the following fields:\
                Natural Language Processing Field\
                or Software Engineering Field\
                or Data and Learning Field\
                or Blockchain Field\
                or Quantum Computing Field\
                or Materials Science or Physics Field

        question: what is the quesiton of the text

        final answer: Extract infomation of the final answer to the question. and give Simplify the answer of the text as the final answer.

        text: {final_answer}

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template=review_template_2)
        messages = prompt.format_messages(final_answer=self.answer,format_instructions=format_instructions)
        # 得到了使用输出解析器的prompt，调用模型
        response = self.llm(messages)
        self.result_dict=output_parser.parse(response.content) 

    def check_is_bad_answer(self):

        answer_schema = ResponseSchema(name="answer",description="\
        The original question is as follows: xx \
        The original answer is as follows: xx \
        The streamlined answer is as follows: xx \
        The optimized question is as follows: xx \
        Whether the optimized question and answer deviate too much from the original question. Answer True if yes,\
                                False if not or unknown.")
    
        response_schemas = [answer_schema]

        # 定义输出解析器
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        # 获得格式化输出
        format_instructions = output_parser.get_format_instructions()
        # 使用输出解析器的提示词
        review_template_2 = """\
        For the following text, extract the following information:

        answer: Whether the optimized question and answer deviate too much from the original question. Answer True if yes,\
                                False if not or unknown.


        original question: {question}

        answer:{answer}
        
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template=review_template_2)
        messages = prompt.format_messages(answer=self.answer,question=self.optimized_question,format_instructions=format_instructions)
        # 得到了使用输出解析器的prompt，调用模型
        response = self.llm(messages)
        if response == None or response.content==None:
            print("can't judge is bad answer")
            return
        self.is_bad = output_parser.parse(response.content) 

    # 预处理问题，翻译并且优化问题
    def preproceed_chain(self):
        user_question_prompt = ChatPromptTemplate.from_template(
            "This is the user's Chinese question. Please translate it into an English question.:"
            "\n\n{Question}"
        )
        chain_translate = LLMChain(llm=self.llm, prompt=user_question_prompt)

        optimize_prompt = ChatPromptTemplate.from_template(
            "Optimize and expand the user's questions, and specify which field the questions belong to. \
            Question:{company_name}"
        )
        chain_optimize = LLMChain(llm=self.llm, prompt=optimize_prompt)
        
        self._preproceed_chain = SimpleSequentialChain(chains=[chain_translate, chain_optimize],verbose=True)
        self.optimized_question = self._preproceed_chain.run(self.original_question)

    def do_answer_question(self,question):
        self.original_question=question
        # 预处理
        self.preproceed_chain()
        print(f"optimized-question:{self.optimized_question}\n")
        # 搜索文档
        self.search_for_doc()
        print(f'abstract:{self.abstract_doc}\n source:{self.source}\n')
        # 路由chain选择领域模版，使用问题相关专业模版回答问题
        self.choose_field_answer_based_doc()
        print(f'row answer:{self.answer}\n')
        # 格式化回答，抽取问题，相关领域，简化回答
        self.format_result()
        print(f'formate answer and extract information to json:\n {self.result_dict}')
        # 检查回答是否满意
        self.check_is_bad_answer()
        print(f'is bad answer:{self.is_bad}')
    
    def translateToChinese(self):
        prompt = ChatPromptTemplate.from_template(
            """Translate the english text to chinese , the output should only about the chinese text:
            <<< TEXT >>>
                {answer}
            <<< TEXT END>>>
            """
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        self.answer = chain.run(self.answer)
        self.optimized_question = chain.run(self.optimized_question)
        self.result_dict['final answer']=chain.run(self.result_dict['final answer'])
    
    def encapsule(self):
        res = {}
        res['original question']=self.original_question
        res['optimized question']=self.optimized_question
        res['question field']=self.result_dict['question field']
        res['answer']=self.answer
        res['simplified answer']=self.result_dict['final answer']
        res['source'] = self.source
        return res

    def answer_question(self,question):
        # 执行回答，如果回答不满意或者偏离问题比较大，执行最多5次重回答
        self.do_answer_question(question)
        count = 0
        while count < 5:
            if self.is_bad==None or self.is_bad['answer']=='True':
                print(f"这是第{count + 1}次重新回答")
                count += 1
                self.do_answer_question(question)
            else:
                print("回答未偏离问题")
                break
        # # 最终的信息判断是否要转回中文
        # if self.language == 'Chinese':
        #     self.translateToChinese()


def read_and_print_input():
    while True:
        user_input = input("请输入问题（输入quit退出）：")
        if user_input == "quit":
            break
        qa = ArXivQA()
        qa.answer_question(question=user_input)
        qa.translateToChinese()
        print("回答结果：")
        print(qa.encapsule())
        print("回答结束，请重新输入问题：")

if __name__ == "__main__":
    read_and_print_input()