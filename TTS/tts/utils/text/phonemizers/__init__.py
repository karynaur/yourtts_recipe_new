from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak

PHONEMIZERS = {ESpeak.name(): ESpeak}


ESPEAK_LANGS = list(ESpeak.supported_languages().keys())

# Dict setting default phonemizers for each language

# Add ESpeak languages and override any existing ones
_ = [ESpeak.name()] * len(ESPEAK_LANGS)
DEF_LANG_TO_PHONEMIZER = dict(list(zip(list(ESPEAK_LANGS), _)))

# Force default for some languages
DEF_LANG_TO_PHONEMIZER ={'cs-cz': 'gruut',
 'lb': 'gruut',
 'ar': 'gruut',
 'fa': 'gruut',
 'fr-fr': 'gruut',
 'sv-se': 'gruut',
 'it-it': 'gruut',
 'sw': 'gruut',
 'en-gb': 'gruut',
 'nl': 'gruut',
 'pt': 'gruut',
 'es-es': 'gruut',
 'ru-ru': 'gruut',
 'en-us': 'gruut',
 'zh-cn': 'zh_cn_phonemizer',
 'de-de': 'gruut',
 'en': 'gruut',
 'ja-jp': 'ja_jp_phonemizer',
 'ko-kr': 'ko_kr_phonemizer',
 'bn': 'bn_phonemizer'}
DEF_LANG_TO_PHONEMIZER["en"] = DEF_LANG_TO_PHONEMIZER["en-us"]


def get_phonemizer_by_name(name: str, **kwargs) -> BasePhonemizer:
    """Initiate a phonemizer by name

    Args:
        name (str):
            Name of the phonemizer that should match `phonemizer.name()`.

        kwargs (dict):
            Extra keyword arguments that should be passed to the phonemizer.
    """
    if name == "espeak":
        return ESpeak(**kwargs)
    raise ValueError(f"Phonemizer {name} not found")


if __name__ == "__main__":
    print(DEF_LANG_TO_PHONEMIZER)
