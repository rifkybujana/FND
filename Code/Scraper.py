from newspaper import Article

def Scrap(url):
    """
    Scrap article from url
    ### Parameter\n
    url : article url (dtype: `string`)\n
    summarize : do you want to summarize the article? (dtype: `boolean`)
    ### Result\n
    return the article text (dtype: `string`)
    """
    
    article = Article(url, language='id')
    article.download()
    article.parse()

    if not article.text or not article.title:
        print("Can't Scrap this article link")
        return None

    return article
