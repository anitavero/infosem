import scrapy


class NewsSpider(scrapy.Spider):
    """Abstract Spider class.
    Usage: scrapy crawl <name> -s JOBDIR=<resume_dir>
    """
    name = "news"
    start_urls = []
    custom_settings = {
        'SCHEDULER_DEBUG': True,
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': 'news.jl'
    }

    # Xpath queries
    xph_title = '//meta[@name="title"]/@content'
    xph_article = '//main/article/*[self::p | self::ul]/descendant-or-self::*/text()'
    xph_author = '//meta[@name="author"]/@content'
    xph_category = '//meta[@name="category"]/@value'
    xph_keywords = '//meta[@name="news_keywords"]/@content'
    xph_description = '//meta[@property="og:description"]/@content'
    xph_date = '//meta[@itemprop="datePublished"]/@content'
    xph_links = '//a/@href'


    def parse(self, response):

        if response.xpath('//meta[@property="og:type"]/@content').extract_first() == 'article':
            yield {
                'title': response.xpath(self.xph_title).extract(),
                'article': response.xpath(self.xph_article).extract(),
                'author': response.xpath(self.xph_author).extract(),
                'category': response.xpath(self.xph_category).extract(),
                'keywords': response.xpath(self.xph_keywords).extract(),
                'description': response.xpath(self.xph_description).extract(),
                'date': response.xpath(self.xph_date).extract(),
                'link': response.url
            }

        next_pages = response.xpath(self.xph_links).extract()
        if next_pages is not None:
            for next_page in next_pages:
                yield response.follow(next_page, self.parse)



class NegySpider(NewsSpider):
    """
    Usage: scrapy crawl negy -s JOBDIR=resume_negy
    """
    name = "negy"
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
    allowed_domains = ['444.hu']
    custom_settings = {
        'SCHEDULER_DEBUG': True,
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': '444.jl'
    }

    xph_links = '//h3/a/@href|//a[@class="infinity-next button"]/@href'


class OrigoSpider(NewsSpider):
    """
    Usage: scrapy crawl origo -s JOBDIR=resume_origo
    """
    name = 'origo'
    start_urls = ['http://www.origo.hu/index.html']
    allowed_domains = ['origo.hu']
    custom_settings = {
        'SCHEDULER_DEBUG': True,
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': 'origo.jl'
    }

    xph_date = '//meta[@name="publish-date"]/@content'
    xph_article = '//article/descendant-or-self::*[self::p | self::ul| self::h2]/descendant-or-self::*/text()'
