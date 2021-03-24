from newspaper import Article

import argparse

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

    article = Article(url)
    article.download()
    article.parse()

    if not article.text:
        print("Can't Scrap this article link")
        return None

    return article.text

if __name__ == "__main__":
    
    ############################################# ARGUMENTS ################################################

    parser = argparse.ArgumentParser(description="This tools is use to scrap article from a given link")
    parser.add_argument('url', type=str, help='url of the article you want to scrap')
    args = parser.parse_args()
    
    ########################################### END ARGUMENTS ##############################################

    text = Scrap(args.url)

    if text:
        print(Scrap(args.url))