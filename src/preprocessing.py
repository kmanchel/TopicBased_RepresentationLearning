import argparse
import os
import time
import utils.xml_utils as xml_utils
import pdb


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    #parser.add_argument("--csv", default=None, type=bool, required=True, help="create csv files from articles")
    #parser.add_argument("--all", default=None, type=bool, required=True, help="convert all files")
    parser.add_argument("--set", default='training', type=str, help="preprocess training or validation")
    parser.add_argument("--dir", type=str, help="Directory for xml input files")
    parser.add_argument("--out", default="semeval_", type=str, help="Output (csv) file")
    parser.add_argument("--lower", default=0, type=int, help="weather to make lowercase or not")
    parser.add_argument("--extra", default=0, type=int, help="whether or not to remove stopwords and lemmatize")
    args = parser.parse_args()

    source_files = []
    truth_files = []

    
    print("Output file format:",args.out)
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if args.set == 'training':
                if file.endswith('xml') and 'validation' not in file:
                    truth_files.append(str(os.path.join(root, file))) if 'ground-truth' in file else source_files.append(
                        str(os.path.join(root, file)))
            else:
                if file.endswith('xml') and 'training' not in file:
                    truth_files.append(str(os.path.join(root, file))) if 'ground-truth' in file else source_files.append(
                        str(os.path.join(root, file)))

    print("Source Files:",source_files)
    print("Truth Files:",truth_files)
    
    print("Lower Case Flag:", bool(args.lower))
    print("Extra Clean Flag:", bool(args.extra))

    xml_utils.parse_xml_files(source_files, truth_files, True, args.dir + "/" + args.out + "processed.csv",  bool(args.lower), bool(args.extra))
    end = time.time()
    print(end - start)