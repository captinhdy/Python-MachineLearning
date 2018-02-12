import scrapy
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from django.utils.encoding import smart_str, smart_unicode

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://finance.google.com/finance/company_news?q=OTCMKTS%3AMJNA'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, cookies=False)

    def parse(self, response):
        page = response.body
        articles = Selector(text=page).xpath('//div[@id="news-main"]').extract()
        print len(articles)
        output = ""
        links = ""
        for article in articles:
            #print(article)
            links += smart_str(article)

        links = Selector(text=links).xpath('//a/@href').extract()
        for link in links:
            output += smart_str(link)
            output += '\n'

        filename = 'news.html'
        with open(filename, 'wb') as f:
            f.write(output)
        self.log('Saved file %s' % filename)

