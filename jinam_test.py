import json
import pickle
from os.path import join
from s2and.data import ANDData
from datetime import datetime

import multiprocessing
import ray

# ray.init()
start_time = datetime.now()

with open("data/production_model.pickle", "rb") as _pickle_file:
	models = pickle.load(_pickle_file)
clusterer = models["clusterer"]
# with open("production_model.pickle", "rb") as _pkl_file:
# 	clusterer = pickle.load(_pkl_file)

# clusterer.n_jobs = n_jobs  # True or False
clusterer.n_jobs = True
use_constraints = False
clusterer.use_cache = True  # not required
clusterer.use_default_constraints_as_supervision = use_constraints  # True or False
clusterer.batch_size = 10000000  # these are for making features

dataset_name = "100k"
parent_dir = "/home/s2and/100k"
signatures = join(parent_dir, f"{dataset_name}_signatures.json")
papers = join(parent_dir, f"{dataset_name}_papers.json")
paper_embeddings = join(parent_dir, f"{dataset_name}_specter.pickle")

# @ray.remote
def generate_anddata(signatures,papers, paper_embeddings):
	anddata = ANDData(
		signatures=signatures,
		papers=papers,
		specter_embeddings=paper_embeddings,
		name="ablation-test",
		mode="inference",
		block_type="original",
		# n_jobs = multiprocessing.cpu_count()
	)
	# anddata = ray.get(temp)
	return anddata

# temp = generate_anddata.remote(signatures,papers,paper_embeddings)
# anddata = ray.get(temp)
anddata = generate_anddata(signatures,papers,paper_embeddings)
print("generated anddata")

end_time1 = datetime.now()
pred_clusters, pred_distance_matrices = clusterer.predict(anddata.get_blocks(), anddata)

with open('temp', 'w') as file:
	file.write(json.dumps(pred_clusters))

print("Total time taken ", datetime.now() - start_time)
print("Data time: ", end_time1 - start_time)
print("Clusterer time: ", datetime.now()-end_time1)
