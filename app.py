import os
import requests
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


DATA_FOLDER = "C:/Users/atcha/RAGpoc/Finance/data"
VECTOR_STORE_PATH = "C:/Users/atcha/RAGpoc/Finance/financial_vectorstore"


STOCK_SYMBOLS = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "AMD (AMD)": "AMD",
    "Alibaba (BABA)": "BABA"
}


def get_stock_trends(stock_symbol):
   
    try:
        stock = yf.Ticker(stock_symbol)
        history = stock.history(period="5d")
        latest_price = history["Close"].iloc[-1]
        trend = "📉 Down" if latest_price < history["Close"].iloc[-2] else "📈 Up"
        return latest_price, trend
    except Exception as e:
        return "N/A", f"Error: {str(e)}"


def get_stock_news():
   
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        top_headlines = newsapi.get_top_headlines(category="business", language="en", country="us")
        articles = top_headlines.get("articles", [])
        return articles[:5]  
    except Exception as e:
        return [{"title": "Error fetching news", "url": "#"}]


def get_market_data(symbol):
    
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data.get("Time Series (Daily)", {})

def load_financial_data():

    loaders = [
        PyPDFLoader(os.path.join(DATA_FOLDER, "214253_1_En_12_Chapter_OnlinePDF.pdf")),
        PyPDFLoader(os.path.join(DATA_FOLDER, "Texas Instruments 2023 Annual Report website.pdf")),
        CSVLoader(os.path.join(DATA_FOLDER, "all-data.csv")),
        TextLoader(os.path.join(DATA_FOLDER, "Sentences_AllAgree.txt"))
    ]
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)



    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embedding_model)
    vector_db.save_local(VECTOR_STORE_PATH)
    return vector_db

def setup_ai():

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())


qa_chain = setup_ai()


st.set_page_config(page_title="📊 AI Financial Analyst", layout="wide")
st.title("📈 AI-Powered Financial Analyst")
st.write("🔍 Get live stock trends, market news, and AI-driven investment insights.")


st.sidebar.header("📈 Select a Stock")
selected_stock = st.sidebar.selectbox("Choose a stock:", list(STOCK_SYMBOLS.keys()))

if selected_stock:
    symbol = STOCK_SYMBOLS[selected_stock]
    price, trend = get_stock_trends(symbol)
    
    st.sidebar.write(f"**💰 Current Price:** ${price}")
    st.sidebar.write(f"**📊 Trend:** {trend}")

 
    market_data = get_market_data(symbol)
    latest_day = list(market_data.keys())[0] if market_data else "N/A"
    latest_close = market_data[latest_day]["4. close"] if latest_day != "N/A" else "N/A"
    
    st.sidebar.write(f"📅 **Latest Close Price ({latest_day}):** ${latest_close}")

    trend_query = f"What is the latest investment insight for {symbol}?"
    trend_response = qa_chain.invoke({"query": trend_query})

  
    trend_text = trend_response.get("result", "")

    st.sidebar.subheader("📢 AI Trend Insights")
    st.sidebar.write(trend_text)


    if "I don't know" in trend_text.lower():
        latest_news = get_stock_news()
        trend_text += f"\n📰 **Latest News:** [{latest_news[0]['title']}]({latest_news[0]['url']})"
    
    st.sidebar.write(trend_text)

st.subheader("📰 Latest Market News")
news_articles = get_stock_news()
for article in news_articles:
    st.markdown(f"📌 **[{article['title']}]({article['url']})**")


query = st.text_input("💡 Ask about stock trends:")

if query:
    response = qa_chain.invoke({"query": query})
    
  
    response_text = response.get("result", "")

    if "I don't know" in response_text.lower():
        selected_stock_news = get_stock_news()
        response_text += f"\n📰 **Latest News:** [{selected_stock_news[0]['title']}]({selected_stock_news[0]['url']})"
    
    st.subheader("📊 AI-Generated Market Insights")
    st.write(response_text)
