from firecrawl import FirecrawlApp

def perform_crawl(url, text_splitter, clean_content, save_scraped_content_to_mongo, scraped_content_collection):
    app = FirecrawlApp(api_key="fc-9da8b1ca1d2149c495e91439d778fea8")
    crawl_params = {
        'crawlerOptions': {
            'excludes': ['blog/*'],
        }
    }
    try:
        crawl_result = app.crawl_url(url, params=crawl_params)
        print("Raw crawl result:", crawl_result)  # Debugging line to show raw crawl result
        if isinstance(crawl_result, list):
            all_cleaned_content = []
            for item in crawl_result:
                content = item.get('content', '')
                print("Raw content:", content)  # Debugging line to show raw content
                cleaned_content = clean_content(content)
                all_cleaned_content.append(cleaned_content)
            combined_content = "\n\n".join(all_cleaned_content)
            save_scraped_content_to_mongo(url, combined_content, scraped_content_collection)
            return combined_content
        else:
            print("Unexpected crawl result format. Expected a list. Got:", type(crawl_result))
            return None
    except Exception as e:
        print(f"Crawl job failed or was stopped. Status: {str(e)}")
        return None
