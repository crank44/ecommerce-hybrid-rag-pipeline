#navighează prin pagini (pagination) și extrage numele produsului, prețul și disponibilitatea.
# scrapy crawl ecommerce_spider -a target_url="http://books.toscrape.com" -o produse.json
import scrapy
from datetime import datetime
from business_intelligence_rag.items import BusinessArticleItem

class EcommerceSpider(scrapy.Spider):
    name = "ecommerce_spider"
    
    # Eliminăm start_urls fix, deoarece îl vom primi din terminal
    def __init__(self, target_url=None, *args, **kwargs):
        super(EcommerceSpider, self).__init__(*args, **kwargs)
        # Salvăm URL-ul primit sau punem unul ca fallback în caz că uităm să îl tastăm
        self.start_urls = [target_url] if target_url else ["http://toscrape.com"]
        
        # Extragem și domeniul automat din URL pentru a-l pune în allowed_domains
        if target_url:
            from urllib.parse import urlparse
            self.allowed_domains = [urlparse(target_url).netloc]
        else:
            self.allowed_domains = ["://toscrape.com"]

    def parse(self, response):
         # Selector actualizat care prinde toate link-urile de produse de pe toscrape.com
        product_links = response.css('h3 a::attr(href)').getall()
        
        # DEBUG: Vedem în terminal dacă Scrapy a găsit vreun link
        self.log(f"=== DETECTATE: {len(product_links)} LINK-URI DE PRODUSE PE ACEASTĂ PAGINĂ ===")

        for link in product_links:
            full_product_url = response.urljoin(link)
            yield response.follow(full_product_url, self.parse_product_details)

        # Navigare automată pe pagina următoare (Pagination)
        next_page = response.css('li.next a::attr(href)').get()
        if next_page:
            full_next_page_url = response.urljoin(next_page)
            yield response.follow(full_next_page_url, callback=self.parse)

    def parse_product_details(self, response):
        item = BusinessArticleItem()
        
        title = response.css('h1::text').get()
        price = response.css('p.price_color::text').get()
        
        # 1. Extragem textul brut
        stock_raw = response.css('p.instock.availability::text').getall()
        
        # 2. Curățăm spațiile și liniile noi pentru fiecare fragment, eliminând elementele goale
        stock_fragments = [fragment.strip() for fragment in stock_raw if fragment.strip()]
        
        # 3. Unim fragmentele rămase printr-un singur spațiu curat
        stock_clean = " ".join(stock_fragments) if stock_fragments else "Unknown"
        
        # Opțional: Dacă tot mai scapă dubluri gen "In stock (19 available) In stock", 
        # păstrăm doar prima parte relevantă:
        if "available" in stock_clean:
            stock_clean = stock_clean.split(")")[0] + ")"

        item['title'] = title.strip() if title else "No Title"
        item['source_url'] = response.url
        
        # Acum content-ul va fi complet liniar, fără \n sau tab-uri
        item['content'] = f"Product: {item['title']}. Price: {price}. Availability: {stock_clean}."
        item['date_extracted'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        yield item
        
        # yield item
