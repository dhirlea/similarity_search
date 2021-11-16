import spacy
import json
import re

def main():

    nlp = spacy.load('en_core_web_sm')

    with open('example_converted_report.txt','r', encoding='utf-8') as f:
        text_example = f.read()

    doc = nlp(text_example)

    parsed_doc = {sent_no:sentence.text for sent_no, sentence in enumerate(doc.sents)}
    cleaned_doc = {}
    i = 0
    for sent_no, sentence in parsed_doc.items():
        cleaned_sentence = parsed_doc[sent_no].replace("\n","").replace("\f","").replace("\t","").replace("•","").strip()
        if re.match("[a-zA-Z0-9]*", cleaned_sentence):
            cleaned_doc[i] = ' '.join([word for word in cleaned_sentence.split() if len(word)>0])
            i +=1 

    with open('parsed_report.json','w',encoding='utf-8') as f:
        f.write(json.dumps(cleaned_doc, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()