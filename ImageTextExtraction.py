from PIL import Image
import numpy as np
import os
import cv2
import glob
import shutil
import pytesseract
import re
import time
import argparse
from statistics import mode
from nltk.corpus import wordnet as wn
import nltk

def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(argument, "Invalid method")

def crop_image(img, start_x, start_y, end_x, end_y):
    cropped = img[start_y:end_y, start_x:end_x]
    return cropped

def get_string(img_path, method, langugage='eng'):
    # Read image using opencv
    img = cv2.imread(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Crop the areas where provision number is more likely present
    #img = crop_image(img, pnr_area[0], pnr_area[1], pnr_area[2], pnr_area[3])
    #img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    #  Apply threshold to get image with only black and white
    img = apply_threshold(img, method)
    save_path = os.path.join(output_path, file_name + "_filter_" + str(method) + ".jpg")
    cv2.imwrite(save_path, img)

    # Set the pytesseract executable path here
    pytesseract.pytesseract.tesseract_cmd = r"E:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang=langugage)

    return result

def find_match(regex, text):
    matches = re.finditer(regex, text, re.MULTILINE)
    target = ""
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1

        print("  Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                            end=match.end(), match=match.group()))
        target = match.group()

    return target

def pretty_print(result_dict):
    s = ''
    for key in result_dict:
        s += '# ' + key + ': ' + result_dict[key] + '\n'
    return s

def isVerbPresent(dict_vocab):
    token_present=False
    valid_list = ['v']
    for w in dict_vocab:
        try:
            tmp = wn.synsets(w)[0].pos()
            if tmp in set(valid_list):
                token_present=True
                break
            #print (w, ":", tmp)
        except:
            print("some error occurred while finding a verb")
    return token_present

def validate_sentence(text):
    #porter_stemmer = PorterStemmer()
    #tokenized_text = nltk.word_tokenize(sentence)
    #sent_length = len(tokenized_text)
    #text_vocab = set(w.lower() for w in text.split() if w.isalpha())
    
    #text_vocab = set(porter_stemmer.stem(w.lower()) for w in nltk.word_tokenize(text) if w.isalpha())

    token_present = False
    text_vocab = set(w.lower() for w in nltk.word_tokenize(text) if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    
    #print(unusual, text_vocab)
    #print(len(unusual)/len(text_vocab))
    if isVerbPresent(text_vocab) == False:
        try:
            if len(unusual)/len(text_vocab) <=0.1:
                return True
            else:
                return True
        except:
            print("Error while division")
    else:
        return True

def clean_sent(text):
    paragraph = ""
    questions = ""
    question = []
    clean_sentences = []
    punctuation='!,:;“”"\')(_-'
    #newstring=text.translate(str.maketrans('', '', punctuation))
    #print("The new sentence# {} is {}".format(1,newstring))
    #sent_text = nltk.sent_tokenize(newstring) # this gives us a list of sentences
    sent_text = text.splitlines()
    # now loop over each sentence and tokenize it separately
    #s_count=0

    whPattern = re.compile(r'who|what|how|where|when|why|which|whom|whose', re.IGNORECASE)

    for sentence in sent_text:
        #s_count = s_count + 1
        #print("The sentence# {} is {}".format(s_count,sentence))
        #print("Is a blank line : {}".format(sentence.strip() == ''))
        if (sentence.strip() != ''):
            '''if whPattern.search(sentence):
                question.append(sentence)
				clean_sentences.append(sentence)
			'''
            if validate_sentence(sentence) == True:
                clean_sentences.append(sentence)
        paragraph = '\n'.join(clean_sentences)
        questions = '\n'.join(question)
    return paragraph, questions

regex = r"P\d{17}"
found = {}
results = {}
queue = []
done = []
missing = []
pnr_area = [150, 450, 1600, 1150]  # [start_x, start_y, end_x, end_y]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program extracts text and questions from a set of documents.")
    parser.add_argument("-i", "--input_dir", help="Input directory for the files to be modified")
    parser.add_argument("-o", "--output_dir", help="Output directory for the files to be modified")
    parser.add_argument("-l", "--language", help="Language present in the image file")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    language = args.language
    
    #exit
    #input_dir ="C:\\Users\\Dell\\" 
    #output_dir = "C:\\Users\\Dell\output\\"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    im_names = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.jpeg'))

    overall_start_t = time.time()
    for im_name in sorted(im_names):
        queue.append(im_name)

    print("The following files will be processed and their provision numbers will be extracted: {}\n".format(queue))  

#f = open("C:\\Users\\Dell\\AutomaticQuestionGenerator\\DB\\db02.txt", "r")
#clean_sent(f.read())

    for im_name in im_names:
        start_time = time.time()
        print("*** The documents that are in the queue *** \n{}\n".format(queue))

        print('#=======================================================')
        print(('# Regex is being applied on {:s}'.format(im_name)))
        print('#=======================================================')
        queue.remove(im_name)
        file_name = im_name.split(".")[0].split("/")[-1]

        i = 1
        while i < 2:
            print("> The filter method " + str(i) + " is now being applied.")
            result = get_string(im_name, i)
            clean_text, question_text = clean_sent(result)
            match = find_match(regex, clean_text)
            if match:
                if file_name in found:
                    found[file_name].append(match)
                else:
                    list = []
                    list.append(match)
                    found[file_name] = list
            #print(output_dir)
            output_path =os.path.join(output_dir, file_name)
            #print(output_path)
            save_path = os.path.join(output_path, file_name + "_paragraph_" + str(i) + ".txt")
            #print(save_path)
            #f = open(os.path.join(output_dir, file_name + "_paragraph_" + str(i) + ".txt"), 'w')
            f = open(save_path, 'w')
            f.write(clean_text)
            f.close()
            #save_path = os.path.join(output_path, file_name + "_questions_" + str(i) + ".txt")
            #f = open(save_path, 'w')
            #f.write(question_text)
            #f.close()
            i += 1

        pnr = ''
        if file_name in found:
            pnr = mode(found[file_name])
            results[file_name] = pnr
            done.append(file_name)
        else:
            missing.append(file_name)
        end_time = time.time()

        print('#=======================================================\n'
              '# Results for: ' + file_name + '\n' 
              '#=======================================================\n' 
              '# The provision number: ' + pnr + '\n'
              '# It took ' + str(end_time-start_time) + ' seconds.     \n'
              '#=======================================================\n')

    overall_end_t = time.time()

    print('#=======================================================\n'
          '# Summary \n'
          '#=======================================================\n'
          '# The documents that are successfully processed are: \n' + pretty_print(results) +
          '#=======================================================\n'
          '# The program failed to extract information from: \n' 
          '# ' + str(missing) + '\n'
          '#=======================================================\n'
          '# It took ' + str(overall_end_t-overall_start_t) + ' seconds.\n'
          '#=======================================================\n')
else:
    print("test")
	
## Use the command as below to invoke the program
## python ImageTextExtraction.py -i "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\Grammar\\" -o "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\Grammar\\output\\"
## python ImageTextExtraction.py -i "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\computer\\" -o "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\computer\\output\\"
## python ImageTextExtraction.py -i "E:\\Sanjaya\\Photos\\ORIYATEST\\" -o "E:\\Sanjaya\\Photos\\ORIYATEST\\output\\" -l "ori"
## python ImageTextExtraction.py -i "E:\\Sanjaya\\Photos\\ORIYATEST\\" -o "E:\\Sanjaya\\Photos\\ORIYATEST\\output\\" -l "eng"

## python ImageTextExtraction.py -i "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\mathematics\\" -o "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\mathematics\\output\\"

## python ImageTextExtraction.py -i "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\English\\term2\\" -o "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\English\\term2\\output\\"

## python ImageTextExtraction.py -i "E:\\Sanjaya\\" -o "E:\\Sanjaya\\output\\"

## python ImageTextExtraction.py -i "C:\\Users\\Dell\\AutomaticQuestionGenerator\\textextract_img\\" -o "C:\\Users\\Dell\\AutomaticQuestionGenerator\\textextract_img\\output\\"

#python ImageTextExtraction.py -i "C:\\Users\\Dell\\AutomaticQuestionGenerator\\test\\" -o "C:\\Users\\Dell\\AutomaticQuestionGenerator\\test\\output\\"

#python ImageTextExtraction.py -i "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\English\\term2\\" -o "E:\\Sanjaya\\Photos\\TOSHI_ENGLISH_CLASS2\\English\\term2\\output\\"

## python ImageTextExtraction.py -i "C:\\Users\\Dell\\AutomaticQuestionGenerator\\images\\" -o "C:\\Users\\Dell\\AutomaticQuestionGenerator\\images\\output\\"