import spacy
import json

def main():

    nlp = spacy.load('en_core_web_sm')

    with open('example_converted_report.txt','r', encoding='utf-8') as f:
        text_example = f.read()

    doc = nlp(text_example)

    parsed_doc = {sent_no:sentence.text for sent_no, sentence in enumerate(doc.sents)}

    with open('parsed_report.json','w',encoding='utf-8') as f:
        f.write(json.dumps(parsed_doc, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()