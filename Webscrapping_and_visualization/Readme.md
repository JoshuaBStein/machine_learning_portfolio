📌 Project Overview
This project is a comparative data analysis tool that visualizes 
the relationship between historical stock prices and revenue for 
two major companies: Tesla (TSLA) and GameStop (GME). By overlaying 
stock performance with quarterly revenue, this dashboard provides 
insights into how market valuation correlates with financial growth.

Key Technical Achievements:
(1). API Integration: Automated extraction of historical stock market data using the yfinance library.
(2). Web Scraping: Developed a custom scraper using BeautifulSoup to extract financial tables from 
HTML documents where direct APIs were unavailable.
(3). Data Engineering: Performed cleaning and normalization on scraped currency strings to 
prepare them for numerical analysis.
(4) Advanced Visualization: Leveraged Plotly to build a dual-axis interactive dashboard 
with range sliders for granular time-series inspection.

🛠️ Tech Stack
Programming: Python 3.11
Data Libraries: Pandas, yfinance
Scraping: BeautifulSoup4, Requests, html5lib
Visualization: Plotly

📂 Project Structure
TeslavsGamestop.py: The core script containing the data pipeline from extraction to visualization.

📊 How It Works
Extraction: The script pulls maximum historical stock data for TSLA and GME.
Scraping: It targets specific AWS-hosted data nodes to retrieve historical revenue records.
Cleaning: Removes symbols like $ and , from revenue strings and converts them to float types for plotting.
Visualization: Generates a 2-row subplot showing the price trend (top) and revenue trend (bottom).

🚀 Getting Started
Prerequisites
Install the required dependencies via pip:

pip install yfinance pandas requests bs4 plotly html5lib

Running the Analysis
Simply execute the Python script:

python TeslavsGamestop.py

Author: Joshua Stein 
Project Context: Developed as part of a Data Science Foundations using R Coursera class,
demonstrating proficiency in end-to-end data pipelines and analysis.
