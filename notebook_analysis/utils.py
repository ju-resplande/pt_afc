from urllib.parse import urlparse
from collections import Counter
from functools import reduce
from statistics import mean
import string
import re

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from tqdm.auto import tqdm
import pandas as pd

# https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/preprocessing.py

# Punctuation list
punctuations = re.escape("!\"#%'()*+,./:;<=>?@[\\]^_`{|}~")

# ##### #
# Regex #
# ##### #
re_remove_brackets = re.compile(r"\{.*\}")
re_remove_html = re.compile(r"<(\/|\\)?.+?>", re.UNICODE)
re_transform_numbers = re.compile(r"\d", re.UNICODE)
re_transform_emails = re.compile(r"[^\s]+@[^\s]+", re.UNICODE)
re_transform_url = re.compile(r"(http|https)://[^\s]+", re.UNICODE)
# Different quotes are used.
re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
re_quotes_3 = re.compile(r"(?u)[‘’`′“”]", re.UNICODE)
re_dots = re.compile(r"(?<!\.)\.\.(?!\.)", re.UNICODE)
re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
re_hiphen = re.compile(r" -(?=[^\W\d_])", re.UNICODE)
re_tree_dots = re.compile("…", re.UNICODE)
# Differents punctuation patterns are used.
re_punkts = re.compile(r"(\w+)([%s])([ %s])" % (punctuations, punctuations), re.UNICODE)
re_punkts_b = re.compile(
    r"([ %s])([%s])(\w+)" % (punctuations, punctuations), re.UNICODE
)
re_punkts_c = re.compile(r"(\w+)([%s])$" % (punctuations), re.UNICODE)
re_changehyphen = re.compile("–")
re_doublequotes_1 = re.compile(r"(\"\")")
re_doublequotes_2 = re.compile(r"(\'\')")
re_trim = re.compile(r" +", re.UNICODE)


def clean_text(text):
    """Apply all regex above to a given string."""
    text = text.lower()
    text = text.replace("\xa0", " ")
    text = re_tree_dots.sub("...", text)
    text = re.sub("\.\.\.", "", text)
    text = re_remove_brackets.sub("", text)
    text = re_changehyphen.sub("-", text)
    text = re_remove_html.sub(" ", text)
    text = re_transform_numbers.sub("0", text)
    text = re_transform_url.sub("URL", text)
    text = re_transform_emails.sub("EMAIL", text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', "", text)
    text = re_dots.sub(".", text)
    text = re_punctuation.sub(r"\1", text)
    text = re_hiphen.sub(" - ", text)
    text = re_punkts.sub(r"\1 \2 \3", text)
    text = re_punkts_b.sub(r"\1 \2 \3", text)
    text = re_punkts_c.sub(r"\1 \2", text)
    text = re_doublequotes_1.sub('"', text)
    text = re_doublequotes_2.sub("'", text)
    text = re_trim.sub(" ", text)
    return text.strip()


STOPWORDS = set(
    """
a à às área acerca ademais adeus agora ainda algo algumas alguns ali além ambas ambos antes
ao aos apenas apoia apoio apontar após aquela aquelas aquele aqueles aqui aquilo
as assim através atrás até aí anos

baixo bastante bem boa bom breve

cada caminho catorze cedo cento certamente certeza cima cinco coisa com como
comprida comprido conhecida conhecido conselho contra contudo corrente cuja
cujo custa cá

da daquela daquele dar das de debaixo demais dentro depois des desde dessa desse
desta deste deve devem deverá dez dezanove dezasseis dezassete dezoito diante
direita disso disse diz dizem dizer do dois dos doze duas durante dá dão

e é és ela elas ele eles em embora enquanto entre então era essa essas esse esses esta
estado estar estará estas estava este estes esteve estive estivemos estiveram
estiveste estivestes estou está estás estão eu eventual exemplo

falta fará favor faz fazeis fazem fazemos fazer fazes fazia faço fez fim final
foi fomos for fora foram forma foste fostes fui

geral grande grandes grupo

inclusive iniciar inicio ir irá isso isto

já

lado lhe ligado local logo longe lugar lá

maior maioria maiorias mais mal mas me meio menor menos meses mesmo meu meus mil
minha minhas momento muito muitos máximo mês milhões mostra

na nada naquela naquele nas nem nenhuma nessa nesse nesta neste no nos nossa
nossas nosso nossos nova novas nove novo novos num numa nunca nuns não nível nós
número números 

o obrigada obrigado oitava oitavo oito onde ontem onze ora os ou outra outras outros

para pra parece parte partir pegar pela pelas pelo pelos perto pode podem poder poderá
podia pois ponto pontos por porquanto porque porquê portanto porém posição
possivelmente posso possível pouca pouco povo primeira primeiro próprio próxima
próximo puderam pôde põe põem

quais qual qualquer quando quanto quarta quarto quatro que quem quer querem quero
questão quieta quieto quinta quinto quinze quê

relação r

sabe saber se segunda segundo sei seis sem sempre ser seria sete seu seus sexta
sexto sim sistema sob sobre sois somente somos sou sua suas são sétima sétimo só

tais tal talvez também tanta tanto tarde te tem temos tempo tendes tenho tens
tentar tentaram tente tentei ter terceira terceiro teu teus teve tipo tive
tivemos tiveram tiveste tivestes toda todas todo todos treze três tu tua tuas
tudo tão têm

um uma umas uns usa usar último url uso

vai vais valor veja vem vens ver vez vezes vinda vindo vinte você vocês vos vossa
vossas vosso vossos vários vão vêm vós vamos

zero
""".split()
)

STOPWORDS.update(set(stopwords.words("portuguese")))


to_remove = [
    "web",
    "www",
    "m",
    "chat",
    "www1",
    "www12",
    "com",
    "org",
    "wordpress",
    "blogspot",
    "net",
    "br",
    "pt",
    "en",
    "blog",
    "info",
    "news",
    "edu",
    "live",
    "site",
    "uk",
    "co",
    "nz",
    "ar",
    "fr",
    "api",
    "it",
    "jor",
    "in",
    "tv",
    "link",
    "cn",
    "online",
    "page",
    "noticias",
    "club",
    "radio",
    "pe",
    "coronavirus",
    "digital",
    "transparencia",
    "cadastros",
    "ba",
]

be_named = [
    "globo",
    "uol",
    "abril",
    "estadao",
    "google",
    "ig",
    "r7",
    "elpais",
    "ebc",
    "apple",
    "painelpolitico",
    "fiocruz",
    "caixa",
    "harvard",
    "sapo",
    "clicrbs",
    "opovo",
    "diariodonordeste",
    "novacruzoficialrn",
    "discord",
    "worldbank",
    "torino",
    "aeiou",
]

gov_br = [
    "jus.br",
    "leg.br",
    "ceeex.eb.mil",
    "dpu.def",
    "marinha.mil",
    "coter.eb.mil",
    ".mp",
    "fab.mil",
    "bdex.eb.mil",
    "25gac.eb.mil",
    "cmp.eb.mil",
    "sgex.eb.mil",
    "hce.eb.mil",
    "4bpe.eb.mil",
    "dfpc.eb.mil",
    "1bpe.eb.mil",
]

gov_usa = ["armyupress.army.mil", "nih.gov", "usembassy.gov", "uscis.gov", "mass.gov"]


def get_domain(data, col="text_urls", drop_mumin=True):
    data_url = dict()
    for dataset in data:
        d_url = reduce(lambda a, b: list(a) + list(b), data[dataset][col])

        domains = list()
        for url in d_url:
            domain = urlparse(url).netloc

            if not domain:
                continue

            if domain == "youtu.be":
                domain = "youtube"
            elif domain in "t.co":
                domain = "twitter"
            elif domain in "t.me":
                domain = "telegram"
            elif domain == "glo.bo":
                domain = "globo"
            elif domain == "flip.it":
                domain = "flipboard"
            elif domain == "hoje.vc":
                domain = "hojeemdia"
            elif domain == "ift.tt":
                domain = "ifttt"
            elif "forms.gle" in domain or "goo.gl" in domain:
                domain = "google"
            elif domain.endswith(".gov") or domain.endswith("armyupress.army.mil"):
                domain = "gov"
            elif "gov" in domain:
                domain = ".".join(domain.split(".")[-2:])
            elif ".usp" in domain:
                domain = "usp"
            elif any(end in domain for end in gov_br):
                domain = "gov.br"
            elif domain == "bit.ly":
                pass
            elif (
                "repositorio" in domain
                or "periodicos" in domain
                or "revistas" in domain
            ):
                domain = domain.split(".")[-2]
            elif domain == "projetos.imd.ufrn.br":
                domain = "ufrn"
            elif "sbt" in domain:
                domain = "sbt"
            elif "l.radios" in domain:
                domain = "radios"
            elif ".afp" in domain:
                domain = "afp"
            else:
                parts = [part for part in domain.split(".") if not part in to_remove]
                domain = ".".join(parts)

                set_ = False
                for part in parts:

                    for sample in be_named:
                        if part == sample:
                            domain = sample
                            set_ = True
                            break

                    if set_:
                        break

            domains.append(domain)

        data_url[dataset] = Counter(domains)

    data_url = pd.DataFrame(data_url).fillna(0).astype(int)
    if drop_mumin:
        data_url.drop(columns="MuMiN-PT")
    data_url["Total"] = data_url.sum(axis=True)
    data_url = data_url.sort_values("Total", ascending=False)

    return data_url


def remove_punc(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def get_word_stats(data, col="text_no_url"):
    data_words = dict()

    data_claim = dict()
    for dataset in tqdm(data):
        data_claim[dataset] = data[dataset][data[dataset][col].notna()]
        data_claim[dataset]["text_cleaned"] = data_claim[dataset][col].apply(clean_text)

        data_claim[dataset]["n_excl"] = data_claim[dataset]["text_cleaned"].str.count(
            "!"
        )
        data_claim[dataset]["has_excl"] = data_claim[dataset]["n_excl"].apply(
            lambda i: int(i > 0)
        )

        words = data_claim[dataset]["text_cleaned"].apply(
            lambda text: re.split(r"\s+", remove_punc(text))
        )

        sents = data_claim[dataset]["text_cleaned"].apply(
            lambda text: sent_tokenize(text, language="portuguese")
        )

        data_words[dataset] = Counter(
            [
                w.lower()
                for ws in words
                for w in ws
                if w not in STOPWORDS and re.search(r"[a-z]", w, flags=re.IGNORECASE)
            ]
        )

        data_claim[dataset]["n_words"] = words.apply(len)
        data_claim[dataset]["avg_word_size"] = words.apply(
            lambda ws: mean(len(w) for w in ws)
        )
        data_claim[dataset]["n_sents"] = sents.apply(len)

        data_claim[dataset]["n_upper_words"] = words.apply(
            lambda words: len([w for w in words if w.isupper()])
        )
        data_claim[dataset]["n_cap_words"] = words.apply(
            lambda words: len(
                [w for w in words if len(w) > 1 and w[0].isupper() and w[1].islower()]
            )
        )

        data_claim[dataset]["n_words_per_sent"] = sents.apply(
            lambda ss: mean(len(re.split(r"\s+", remove_punc(s))) for s in ss)
        )

    return data_claim, data_words
