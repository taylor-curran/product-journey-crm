# test_query_turbopuffer.py

from raggy.vectorstores.tpuf import TurboPuffer


with TurboPuffer(namespace='test-tay') as t:
    result = t.query("What is the best doc to use when I want a Prefect quickstart?", top_k=2)


    # result.data[0].attributes['text']
    for i in range(len(result.data)):
        print('--------------------------')
        print(result.data[i].attributes['text'])


# For Rag App use below

# from raggy.vectorstores.tpuf import query_namespace, TurboPuffer


# result = query_namespace("How should I begin using Prefect?", namespace="test-tay", top_k=2)

# print(result)
