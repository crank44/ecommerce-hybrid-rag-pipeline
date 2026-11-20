import scrapy

class BusinessArticleItem(scrapy.Item):
    title = scrapy.Field()       # Titlul produsului sau al articolului de business
    source_url = scrapy.Field()  # URL-ul de unde am luat informația
    content = scrapy.Field()     # Textul complet/descrierea (care va merge în RAG)
    date_extracted = scrapy.Field() # Data extragerii (importantă pentru relevanța business)