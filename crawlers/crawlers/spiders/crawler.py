import scrapy


class NewsSpider(scrapy.Spider):
    name = "news"
    start_urls = [
        'https://444.hu/author/acsd/',
        'https://444.hu/author/alberta/',
        'https://444.hu/author/bedem/',
        'https://444.hu/author/borosj/',
        'https://444.hu/author/botost/',
        'https://444.hu/author/czinkoczis/',
        'https://444.hu/author/erdelyip/',
        'https://444.hu/author/halaszj/',
        'https://444.hu/author/haszanz/',
        'https://444.hu/author/herczegm/',
        'https://444.hu/author/horvathb/',
        'https://444.hu/author/akiraly/',
        'https://444.hu/author/kulcsarrebeka/',
        'https://444.hu/author/magyarip/',
        'https://444.hu/author/plankog/',
        #https://'444.hu/category/hirdetes/',
        'https://444.hu/author/renyip/',
        'https://444.hu/author/sarkadizs/',
        'https://444.hu/author/szily/',
        'https://444.hu/author/tbg/',
        'https://444.hu/author/peteru/',
        'https://444.hu/author/urfip/',
        'https://444.hu/author/vajdag/'
    ]
    data_file = '444.jl'
    custom_settings = {
        'SCHEDULER_DEBUG': True,
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': data_file
    }

    def parse(self, response):

        if response.xpath('//meta[@property="og:type"]/@content').extract_first() == 'article':
            yield {
                'title': response.xpath('//*[@id="headline"]/h1/text()').extract(),
                'article': response.xpath('//main/article/*[self::p | self::ul]/descendant-or-self::*/text()').extract(),
                'author': response.xpath('//meta[@name="author"]/@content').extract(),
                'category': response.xpath('//meta[@name="category"]/@value').extract(),
                'keywords': response.xpath('//meta[@name="news_keywords"]/@content').extract(),
                'description': response.xpath('//meta[@name="description"]/@content').extract(),
                'date': response.xpath('//meta[@itemprop="datePublished"]/@content').extract(),
                'link': response.url
            }

        next_pages = response.xpath('//h3/a/@href|//a[@class="infinity-next button"]/@href').extract()
        if next_pages is not None:
            for next_page in next_pages:
                yield response.follow(next_page, self.parse)
