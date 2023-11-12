import deepl

from config.env_reader import TranslatorConfig

translator = deepl.Translator(TranslatorConfig.deepl_auth_key)

def translate_to_korean(text:str)->str:
    result = translator.translate_text(text,target_lang="KO")
    return result.text

