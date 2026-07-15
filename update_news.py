import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
# import os

def fetch_team_news(team_name):
    query = f'"{team_name}" football (injury OR "ruled out" OR "fitness test" OR transfer OR "swap deal" OR "squad change" OR "in form" OR "top form" OR "peak condition") when:14d'
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        titles = []
        # Find all item elements and extract their titles
        for item in root.findall('./channel/item'):
            title = item.find('title')
            if title is not None:
                titles.append(title.text)
                
        # Join top 5 latest headlines
        news_summary = " | ".join(titles[:5])
        return news_summary if news_summary else "No recent news found."
    except Exception as e:
        print(f"Error fetching news for {team_name}: {e}")
        return "Error fetching news."

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python update_news.py <team1> <team2> ...")
        sys.exit(1)
        
    teams = sys.argv[1:]
    csv_path = './context/team_news.csv'
    
    print(f"Creating/updating {csv_path} with teams: {', '.join(teams)}")
    
    data = []
    print("Fetching news for teams:")
    for team in teams:
        print(f" - {team}")
        news = fetch_team_news(team)
        data.append({"team": team, "recent_form": news})
        
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print("Updated team_news.csv successfully!")

if __name__ == "__main__":
    main()
