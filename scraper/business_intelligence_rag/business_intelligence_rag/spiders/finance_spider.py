#extrage titlurile și link-urile articolelor de business de pe prima pagină Reuters.
# scrapy crawl finance_spider -a target_url="https://npr.org" -o stiri.csv
import scrapy
from datetime import datetime
from business_intelligence_rag.items import BusinessArticleItem

class FinanceSpider(scrapy.Spider):
    name = "finance_spider"

    # Păstrăm doar setările de bază, fără SPIDER_MIDDLEWARES care dădea crash
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'DOWNLOAD_DELAY': 1,
    }

    def __init__(self, target_url=None, *args, **kwargs):
        super(FinanceSpider, self).__init__(*args, **kwargs)
        self.start_urls = [target_url if target_url else "https://npr.org"]
        self.allowed_domains = ["npr.org"]

    def parse(self, response):
        all_links = response.css('a::attr(href)').getall()
        
        article_links = []
        for link in all_links:
            if "/2026/" in link or "sections/business" in link:
                full_url = response.urljoin(link)
                article_links.append(full_url)
        
        unique_articles = list(set(article_links))
        
        self.logger.info(f"=== URL EXECUTAT: {response.url} ===")
        self.logger.info(f"=== ARTICOLE REALE DETECTATE: {len(unique_articles)} ===")

        # Luăm primele 5 articole pentru test
        for full_url in unique_articles[:5]:
            # dont_filter=True dezactivează verificarea offsite nativ, pe fiecare request
            yield scrapy.Request(full_url, callback=self.parse_article_details, dont_filter=True)

    def parse_article_details(self, response):
        if not hasattr(response, "text"):
            return

        item = BusinessArticleItem()
        title = response.css('h1::text').get() or response.css('.storytitle h1::text').get()
        
        paragraphs_raw = response.css('#storytext p::text').getall() or response.css('article p::text').getall() or response.css('p::text').getall()
        paragraphs_clean = [p.strip() for p in paragraphs_raw if p.strip()]
        content_clean = " ".join(paragraphs_clean)
        
        if title and len(content_clean) > 200:
            item['title'] = title.strip()
            item['source_url'] = response.url
            item['content'] = content_clean
            item['date_extracted'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.logger.info(f"-> Trimit la vectorizare: {item['title']}")
            yield item
