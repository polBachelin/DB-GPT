import os
from typing import Any, Dict, Optional

from pandas import DataFrame
from pydantic import BaseModel, Field

from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH
from dbgpt.core import LLMClient, ModelMessage, ModelMessageRoleType, ModelRequest
from dbgpt.core.awel import DAG, HttpTrigger, JoinOperator, MapOperator
from dbgpt.datasource.rdbms.base import RDBMSDatabase
from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnect
from dbgpt.datasource.rdbms.conn_postgresql import PostgreSQLDatabase
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt.rag.embedding.embedding_factory import DefaultEmbeddingFactory
from dbgpt.rag.operators.schema_linking import SchemaLinkingOperator
from dbgpt.storage.vector_store.chroma_store import ChromaVectorConfig
from dbgpt.storage.vector_store.connector import VectorStoreConnector
from dbgpt.util.chat_util import run_async_tasks
import logging
from dbgpt.storage.vector_store.base import VectorStoreConfig
from dbgpt.serve.rag.assembler.db_schema import DBSchemaAssembler
from dbgpt.rag.embedding.embedding_factory import DefaultEmbeddingFactory

logger = logging.getLogger(__name__)

"""AWEL: Simple nl-schemalinking-sql-chart operator example

    pre-requirements:
        1. install openai python sdk
        ```
            pip install "db-gpt[openai]"
        ```
        2. set openai key and base
        ```
            export OPENAI_API_KEY={your_openai_key}
            export OPENAI_API_BASE={your_openai_base}
        ```
        or
        ```
            import os
            os.environ["OPENAI_API_KEY"] = {your_openai_key}
            os.environ["OPENAI_API_BASE"] = {your_openai_base}
        ```
        python examples/awel/simple_nl_schema_sql_chart_example.py
    Examples:
        ..code-block:: shell
        curl --location 'http://127.0.0.1:5555/api/v1/awel/trigger/examples/rag/schema_linking' \
--header 'Content-Type: application/json' \
--data '{"query": "Statistics of user age in the user table are based on three categories: age is less than 10, age is greater than or equal to 10 and less than or equal to 20, and age is greater than 20. The first column of the statistical results is different ages, and the second column is count."}' 
"""

INSTRUCTION = (
    "I want you to act as a SQL terminal in front of an example database, you need only to return the sql "
    "command to me.Below is an instruction that describes a task, Write a response that appropriately "
    "completes the request.\n###Instruction:\n{}"
)
INPUT_PROMPT = "\n###Input:\n{}\n###Response:"


def _create_vector_connector():
    """Create vector connector."""
    return VectorStoreConnector.from_default(
        "Chroma",
        vector_store_config=ChromaVectorConfig(
            name="vector_name",
            persist_path=os.path.join(PILOT_PATH, "data"),
        ),
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),
    )


def _prompt_join_fn(query: str, chunks: str) -> str:
    prompt = INSTRUCTION.format(chunks + INPUT_PROMPT.format(query))
    print(f"Prompt join fn prompt : {prompt}")
    return prompt


class TriggerReqBody(BaseModel):
    query: str = Field(..., description="User query")


class RequestHandleOperator(MapOperator[TriggerReqBody, Dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, input_value: TriggerReqBody) -> Dict:
        params = {
            "query": input_value.query,
        }
        print(f"Receive input value: {input_value.query}")
        return params


class SqlGenOperator(MapOperator[Any, Any]):
    """The Sql Generation Operator."""

    def __init__(self, llm: Optional[LLMClient], model_name: str, **kwargs):
        """Init the sql generation operator
        Args:
           llm (Optional[LLMClient]): base llm
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._model_name = model_name

    async def map(self, prompt_with_query_and_schema: str) -> str:
        """generate sql by llm.
        Args:
            prompt_with_query_and_schema (str): prompt
        Return:
            str: sql
        """

        messages = [
            ModelMessage(
                role=ModelMessageRoleType.SYSTEM, content=prompt_with_query_and_schema
            )
        ]
        request = ModelRequest(model=self._model_name, messages=messages)
        tasks = [self._llm.generate(request)]
        output = await run_async_tasks(tasks=tasks, concurrency_limit=1)
        sql = output[0].text
        return sql


class SqlExecOperator(MapOperator[Any, Any]):
    """The Sql Execution Operator."""

    def __init__(self, connection: Optional[RDBMSDatabase] = None, **kwargs):
        """
        Args:
            connection (Optional[RDBMSDatabase]): RDBMSDatabase connection
        """
        super().__init__(**kwargs)
        self._connection = connection

    def map(self, sql: str) -> DataFrame:
        """retrieve table schemas.
        Args:
            sql (str): query.
        Return:
            str: sql execution
        """
        dataframe = self._connection.run_to_df(command=sql, fetch="all")
        print(f"sql data is \n{dataframe}")
        return dataframe


class ChartDrawOperator(MapOperator[Any, Any]):
    """The Chart Draw Operator."""

    def __init__(self, **kwargs):
        """
        Args:
        connection (RDBMSDatabase): The connection.
        """
        super().__init__(**kwargs)

    def map(self, df: DataFrame) -> str:
        """get sql result in db and draw.
        Args:
            sql (str): str.
        """
        import matplotlib.pyplot as plt

        category_column = df.columns[0]
        count_column = df.columns[1]
        plt.figure(figsize=(8, 4))
        plt.bar(df[category_column], df[count_column])
        plt.xlabel(category_column)
        plt.ylabel(count_column)
        plt.show()
        return str(df)


with DAG("hard_nl_schema_sql_chart_example") as dag:
    trigger = HttpTrigger(
        "/examples/rag/to_chart", methods="POST", request_body=TriggerReqBody
    )
    request_handle_task = RequestHandleOperator()
    query_operator = MapOperator(lambda request: request["query"])
    llm = OpenAILLMClient()
    model_name = "gpt-3.5-turbo"
    connect = PostgreSQLDatabase.from_uri_db(
        "172.23.0.1", "5432", "postgres", "postgres", "Adventureworks"
    )
    print(f"connect: {connect}")
    print(f" table names: {connect.get_table_names()}")
    embedding_factory = DefaultEmbeddingFactory()
    embedding_fn = embedding_factory.create(
        model_name="/home/polo/github/DB-GPT/models/text2vec-large-chinese"
    )
    vector_name = "Adventureworks_profile"
    config = ChromaVectorConfig(name=vector_name, embedding_fn=embedding_fn)
    vector_store_connector = VectorStoreConnector(
        vector_store_type="Chroma",
        vector_store_config=config,
    )
    retriever_task = SchemaLinkingOperator(
        connection=connect,
        llm=llm,
        model_name=model_name,
        vector_store_connector=vector_store_connector,
    )
    prompt_join_operator = JoinOperator(combine_function=_prompt_join_fn)
    sql_gen_operator = SqlGenOperator(llm=llm, model_name=model_name)
    sql_exec_operator = SqlExecOperator(connection=connect)
    draw_chart_operator = ChartDrawOperator(connection=connect)
    trigger >> request_handle_task >> query_operator >> prompt_join_operator
    (
        trigger
        >> request_handle_task
        >> query_operator
        >> retriever_task
        >> prompt_join_operator
    )
    prompt_join_operator >> sql_gen_operator >> sql_exec_operator >> draw_chart_operator

if __name__ == "__main__":
    if dag.leaf_nodes[0].dev_mode:
        # Development mode, you can run the dag locally for debugging.
        from dbgpt.core.awel import setup_dev_environment

        setup_dev_environment(
            [dag],
            port=5555,
            logging_level="info",
            logger_filename="./dag_output.log",
            show_dag_graph=False,
        )
    else:
        pass
