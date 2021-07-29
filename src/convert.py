import csv 
import json 
import sys

def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
      
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            
            #add this python dict to json array
            jsonArray.append(row)
    
    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, ensure_ascii=False, indent=4)
        jsonResult = "{\"results\": {\n  \"audio_result\":" + jsonString + "}}"
        jsonf.write(jsonResult)
        print(jsonResult)
        
def convert_time(inputcsvpath, outputcsvpath):    
    import time
    import csv
    with open(inputcsvpath, 'r', encoding='utf-8') as input:
        lines = []
        csvReader = csv.reader(input)
        lines.append(next(csvReader))
        for line in csvReader:
            #print(line[0])
            #print(line[1])
            #print(line[2])
            #tmp = format(float(line[1])*100%100, "2.0f")
            #print(tmp)
            #print(format(float(line[1])-float(int(line[1]), ".2f")))
            line[1] = time.strftime('%H:%M:%S', time.gmtime(float(line[1]))) + ':' + format(int(float(line[1])*100%100), "02")
            line[2] = time.strftime('%H:%M:%S', time.gmtime(float(line[2]))) + ':' + format(int(float(line[2])*100%100), "02")
            lines.append(line)
    #print(lines)        
    with open(outputcsvpath, 'w', encoding='utf-8') as output:
        cw = csv.writer(output)
        cw.writerows(lines)