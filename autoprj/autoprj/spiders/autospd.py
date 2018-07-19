# -*- coding: utf-8 -*-
import scrapy
from autoprj.items import AutoprjItem
from scrapy.http import Request

class AutospdSpider(scrapy.Spider):
    name = 'autospd'
    allowed_domains = ['dangdang.com']
    start_urls = ['http://category.dangdang.com/pg1-cid10010513.html']

    def parse(self, response):
        item=AutoprjItem()

        item["name"]=response.xpath("//a[@name='itemlist-title']/@title").extract()
        item["price"]=response.xpath("//spane[@class='price_n']").extract()
        item["link"]=response.xpath("//a[@name='itemlist-title']/@href").extract()
        item["comnum"]=response.xpath("//a[@name='itemlist-review']/text()").extract()
        yield item

        for i in range(1,4):
            url="http://category.dangdang.com/pg"+str(i)+"-cid10010513.html"
            yield Request(url,callback=self.parse)


        #pass
