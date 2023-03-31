import os
import pickle

from recbole.quick_start import load_config

def get_model_file(dataset):
	found = []
	files = sorted([f for f in os.listdir("./bestparam/") if "pickle" in f and dataset in f])
	for file in files:
		#get best param
		with open("./bestparam/"+file, "rb") as f:
			try:
				bestparam = pickle.load(f)
			except:
				f.seek(0)
				bestparam  = pickle.load(f)

		#find which model
		model = file.split("_")[-3]

		#find which file
		candidate = reversed(sorted([f for f in os.listdir("./saved") if model in f])) 
		#reverse to get more recent models
		for model_file in candidate:
			config  = load_config("./saved/"+model_file)
			if config.dataset != dataset:
				continue
			else:
				to_check = [config.final_config_dict[key] for key, _ in bestparam.items()]
				bestparam_val = list(bestparam.values())
				bestparam_val = [x if type(x)!=str else eval(x) for x in bestparam_val]
				if to_check == bestparam_val:
					print(f"found {model}, continue for another best model")
					found.append(model_file)
					break
	return found


list_dataset = [
            "Amazon_Luxury_Beauty",
            "lastfm", 
            "ml-1m",
            "Amazon_Industrial_and_Scientific",
            "book-crossing",
            "Amazon_Digital_Music"
            ]

for dataset in list_dataset:

    print(f"Finding {dataset}")
    try:
        with open(f"results/filename_best_model_for_{dataset}.pickle","rb") as f:
            found = pickle.load(f)
            print("found existing best file")
            print(found)
    except:
        print("no existing file found")
        found = get_model_file(dataset)
        with open(f"results/filename_best_model_for_{dataset}.pickle","wb") as f:
            print("Saving new best file")
            pickle.dump(found, f, pickle.HIGHEST_PROTOCOL)
            print(found)