def test_imports():
    import langchain
    import faiss
    import mistralai
    import pandas
    import requests
    import dotenv
    import tiktoken

    assert True  # le test échoue si un import plante


def test_versions():
    import langchain
    import faiss
    import pandas
    import requests
    import tiktoken

    assert hasattr(langchain, "__version__")
    assert hasattr(faiss, "__version__")
    assert hasattr(pandas, "__version__")
    assert hasattr(requests, "__version__")
    assert hasattr(tiktoken, "__version__")

    print("LangChain :", langchain.__version__)
    print("Faiss :", faiss.__version__)
    print("pandas :", pandas.__version__)
    print("requests :", requests.__version__)
    print("tiktoken :", tiktoken.__version__)