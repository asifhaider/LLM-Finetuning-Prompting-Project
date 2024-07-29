import argparse
import os
import json
from rank_bm25 import BM25Okapi
from time import sleep
import numpy as np
import random
import openai
from smooth_bleu import bleu_fromstr
import traceback

def makestr(lst):
    #p=""
    #for w in lst:
        #p=p+w+" "
    return lst#p.strip()  

# for review comment generation task
def process_code(code):
    # mimics the SimpleGenDataset -> __init__ and/or convert_examples_to_features function
    code = code.split("\n")[1:] # remove start @@
    code = [line for line in code if len(line.strip()) > 0] # remove empty lines
    map_dic = {"-": 0, "+": 1, " ": 2}
    def f(s):
        if s in map_dic:
            return map_dic[s]
        else:
            return 2
    labels = [f(line[0]) for line in code]
    code = [line[1:].strip() for line in code]
    inputstr = ""
    for label, line in zip(labels, code):
        if label == 1:
            inputstr += "<add>" + line
        elif label == 0:
            inputstr += "<delete>" + line
        else:
            inputstr += "<keep>" + line

    return inputstr


# for code refinement generation task
def process_code(code):
    # mimics the RefineDataset -> __init__ and/or tokenize function
    code = code.split("\n")
    code = [line[1:].strip() for line in code]
    code = " ".join(code)
    return code


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--open_key", default=None, type=str, required=True,
                        help="Enter API key")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="davinci/cushman/instruct") # instruct model option added
    #parser.add_argument("--data_folder", default=None, type=str, required=True,
                        #help="data folder path ")
    parser.add_argument("--pause_duration", default=None, type=str, required=True,
                        help="time to stop between samples") 
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="random/BM25")  
    parser.add_argument("--number_of_fewshot_sample", default=None, type=str, required=True,
                        help="1,2,4,6,8") 
    # parser.add_argument("--language", default=None, type=str, required=True,
    #                     help="csharp/cpp")
    parser.add_argument("--type", type=str, required=True, help="Without/Both/Summary/Callgraph")
    parser.add_argument("--length", default=500, type=int, required=True, help="data size") # dataset length option added
    parser.add_argument("--train_file", default=None, type=str, required=True, help="train file path")
    parser.add_argument("--test_file", default=None, type=str, required=True, help="test file path")
    args = parser.parse_args()
    
    
    openai.api_key = args.open_key
    
    
    if args.model=="davinci":
        target_model="code-davinci-002"
    elif args.model=="cushman":
        target_model="code-cushman-001"
    elif args.model=="instruct": # turbo instruct model option added
        target_model="gpt-3.5-turbo-instruct"
    
    # target_model = "gpt-3.5-turbo-16k-0613"

    #Reading data
    train_json = []
    count = 1
    train_filename = args.train_file
    for line in open(train_filename, 'r', encoding="utf-8"):
        train_json.append(json.loads(line))
        print(count)
        count=count+1
    print(len(train_json))
    
    test_json = []
    test_filename = args.test_file
    for line in open(test_filename, 'r', encoding="utf-8"):
        test_json.append(json.loads(line))
    print(len(test_json)) 
    
    # for review comment generation task

    # train_code=[]
    # train_nl=[]
    # train_summary=[]
    # train_callgraph=[]
    # for i in range(len(train_json)):
    #     train_code.append(process_code(train_json[i]['patch']))
    #     train_nl.append(makestr(train_json[i]['msg']))    
    #     train_summary.append(makestr(train_json[i]['summary']))
    #     train_callgraph.append(makestr(train_json[i]['callgraph']))    
        
    # test_code=[]
    # test_nl=[]
    # test_summary=[]
    # test_callgraph=[]
    # for i in range(len(test_json)):
    #     test_code.append(process_code(test_json[i]['patch']))
    #     test_nl.append(makestr(test_json[i]['msg']))   
    #     test_summary.append(makestr(test_json[i]['summary']))
    #     test_callgraph.append(makestr(test_json[i]['callgraph'])) 


    # for code refinement generation task

    train_code_old=[]
    train_nl=[]
    train_summary=[]
    train_callgraph=[]
    train_code_new = []
    for i in range(len(train_json)):
        train_code_old.append(process_code(train_json[i]['old']))
        train_nl.append(makestr(train_json[i]['comment']))    
        train_summary.append(makestr(train_json[i]['summary']))
        train_callgraph.append(makestr(train_json[i]['callgraph']))  
        train_code_new.append(process_code(train_json[i]['new']))  
        

    test_code_old=[]
    test_nl=[]
    test_summary=[]
    test_callgraph=[]
    test_code_new=[]
    for i in range(len(test_json)):
        test_code_old.append(process_code(test_json[i]['old']))
        test_nl.append(makestr(test_json[i]['comment']))   
        test_summary.append(makestr(test_json[i]['summary']))
        test_callgraph.append(makestr(test_json[i]['callgraph'])) 
        test_code_new.append(process_code(test_json[i]['new']))

    
    print(args.mode)
    if args.mode=="BM25":
        tokenized_corpus = [doc.split(" ") for doc in train_code_old]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        indices=[]
        for i in range(len(train_code_old)):
            indices.append(i)
     
    i=0
    is_error=0
    error_count=0
    all_golds=[]
    all_preds=[]
    flog=open('result/'+str(args.length)+"-"+args.type+"-"+'log.txt',"a", encoding="utf-8")
    while i < args.length:#len(test_code):   
        print(i)
        flog.write("Entry "+str(i)+"\n")
        try:
            query = test_code_old[i]
            
            if is_error==0:
                tokenized_query = query.split(" ")
                if args.mode=="BM25":
                    x=bm25.get_scores(tokenized_query)   
                    arr = np.array(x)
                    x=arr.argsort()[-int(args.number_of_fewshot_sample):][::-1]
                    print("in BM25")
                else:
                    random.shuffle(indices)
                    x=indices[0:int(args.number_of_fewshot_sample)]
            
            print(x)
            flog.write(str(x)+"\n")
            
            if is_error==1 and len(x)>1:
                x=x[0:len(x)-1]
                is_error=0
            
            context=""
            for w in x:
                # for review comment generation task 

                # context=context+"Code To Be Reviewed: \t" + train_code[w]+"\n"
                # context=context+"Summary: \t"+train_summary[w]+"\n"
                # context=context+"Callgraph: \t"+train_callgraph[w]+"\n"
                # context=context+"Review Comment: \t<s> "+train_nl[w]+" </s>"+"\n\n"
                
                # for code refinement generation task

                context=context+"[Before Refinement]: \t" + train_code_old[w]+"\n"
                context=context+"[Summary]: \t"+train_summary[w]+"\n"
                context=context+"[Callgraph]: \t"+train_callgraph[w]+"\n"
                context=context+"[Review Comment]: \t" + train_nl[w]+"\n"
                context=context+"[After Refinement]: \t<s> "+train_code_new[w]+" </s>"+"\n\n"
            
            # for review comment generation task

            # context=context+"Code To Be Reviewed: \t"+test_code[i]+"\n"
            # context=context+"Summary: \t"+test_summary[i]+"\n"
            # context=context+"Callgraph: \t"+test_callgraph[i]+"\n"
            # context=context+"Review Comment: \t<s> "

            # for code refinement generation task

            context=context+"[Before Refinement]: \t" + test_code_old[i]+"\n"
            context=context+"[Summary]: \t"+test_summary[i]+"\n"
            context=context+"[Callgraph]: \t"+test_callgraph[i]+"\n"
            context=context+"[Review Comment]: \t" + test_nl[i]+"\n"
            context=context+"[After Refinement]: \t<s> "
            
            print("Context (Prompt + Query):\n" + context)
            flog.write("Context (Prompt + Query):\n" + context+"\n")
            response = openai.Completion.create(
              engine=target_model,
              prompt=context,
              temperature=0.5,
              max_tokens=250, #250
              stop=["</s>"],
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0,
              n=5
            )
            print("##################### Model Response ###############")
            flog.write("##################### Model Response ###############\n")
            print(f"Number of Responses: {len(response.choices)}")
            flog.write(f"Number of Responses: {len(response.choices)}\n")
            modelout = ""
            bleu_score = -1
            
            for j in range(len(response.choices)):
                print(f"Response {j+1}:\n" + response.choices[j].text)
                flog.write(f"Response {j+1}:\n" + response.choices[j].text+"\n")
                resp = response.choices[j].text.split("</s>")[0]
                current_bleu = bleu_fromstr([resp], [test_code_new[i]], rmstop=False)
                print("Current BLEU: ", current_bleu)
                flog.write("Current BLEU: "+str(current_bleu)+"\n")
                if current_bleu > bleu_score:
                    bleu_score = current_bleu
                    modelout = resp

            print(f"Best BLEU: {bleu_score}")
            flog.write(f"Best BLEU: {bleu_score}\n")
            print(f"Final Response:\n" + modelout)
            flog.write(f"Final Response:\n" + modelout+"\n")
            flog.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                        
            fr=open('result/'+str(args.length)+"-"+args.type+"-"+args.model+"_"+args.mode+"_"+args.number_of_fewshot_sample+"_call.txt","a", encoding="utf-8")
            fg=open('result/'+str(args.length)+"-"+args.type+"-"+"gold"+".txt","a",encoding="utf-8")   
            fr.write(' '.join(line.strip() for line in modelout.split('\n')) + "\n")
            fg.write(test_code_new[i].strip()+"\n")
            fr.close()
            fg.close()
            
            print("going_sleep")
            flog.write("going_sleep\n")
            sleep(int(args.pause_duration))
            print("wakeup")
            flog.write("wakeup\n")

            all_golds.append(test_code_new[i])
            all_preds.append(modelout)

            i=i+1
            error_count=0
            is_error=0
            
            
        except:
            print("error")
            flog.write("error\n")
            # print the error traceback
            traceback.print_exc()
            flog.write(traceback.format_exc())

            is_error=1
            error_count=error_count+1
            print(error_count)
            flog.write(str(error_count)+"\n")
            if error_count==10: # change this value to 5 from 10
                 fr=open('result/'+str(args.length)+"-"+args.type+"-"+args.model+"_"+args.mode+"_"+args.number_of_fewshot_sample+"_error_call.txt","a", encoding="utf-8")
                 fg=open('result/'+"gold"+".txt","a",encoding="utf-8") 
                 fr.write(str(i)+"\t"+""+"\n")
                 fr.close()
                 fg.close()
                 i=i+1
                 error_count=0
                 is_error=0
            sleep(30)
            continue
    flog.close()
    bleu = bleu_fromstr(all_preds, all_golds, rmstop=False)
    print("BLEU: ", bleu)
    
    # save the bleu score
    fb=open('result/'+'bleu_scores.txt',"a", encoding="utf-8")
    fb.write(args.type + " " + str(args.length)+"\t")
    fb.write(str(bleu)+"\n")
    fb.close()
    
main()