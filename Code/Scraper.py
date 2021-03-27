from newspaper import Article

import argparse

"""
This script can be used to scrap article from a given link

Author: Rifky Bujana Bisri
email : rifkybujanabisri@gmail.com
"""

lang_id = {
    'Bahasa': 'id',
    'English': 'en'
}

def Scrap(url, lang):
    """
    Scrap article from url

    ### Parameter\n
    url : article url (dtype: `string`)\n
    summarize : do you want to summarize the article? (dtype: `boolean`)

    ### Result\n
    return the article text (dtype: `string`)
    """

    if not lang in lang_id:
        print('language not availabel\nLanguage: {}'.format(list(lang_id.keys())))

    article = Article(url, language=lang_id[lang])
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
    parser.add_argument('lang', type=str, help='article language, [Bahasa, English], default: `Bahasa`', default='Bahasa')
    args = parser.parse_args()
    
    ########################################### END ARGUMENTS ##############################################

    text = Scrap(args.url, args.lang)

    if text:
        print(text)