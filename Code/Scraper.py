from newspaper import Article

"""
This script can be used to scrap article from a given link

Author: Rifky Bujana Bisri
email : rifkybujanabisri@gmail.com
"""

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

    if not article.text:
        print("Can't Scrap this article link")
        return None

    return article.text