from typing import List, Tuple

import torch.nn.functional as F


class Postprocessing:

    # calculate loss
    @staticmethod
    def compute_cosine_similarity(predictions):
        cosine_similarity = []
        for p in predictions:
            emb0 = p[0, 0:64]
            emb1 = p[0, 64:]
            emb0 = emb0.view(1, -1)
            emb1 = emb1.view(1, -1)
            cos_sim = F.cosine_similarity(emb0, emb1)
            cosine_similarity.append(float(cos_sim[0]))
        return cosine_similarity

    @staticmethod
    def get_similarities(dataloader):
        # calculate similarity
        similarities = []
        for batch in dataloader:
            # similarities.append(float(b['similarity'][0]))
            sim_temp = [float(b) for b in batch["similarity"]]
            # similarities = similarities  +  [b for b in batch]
            similarities = similarities + sim_temp
        return similarities

    @staticmethod
    def get_similarities_multitasking(
        dataloader,
    ) -> Tuple[List[float], List[float]]:
        # calculate similarity
        # similarities1 = []
        # similarities2 = []
        # for batch in dataloader:
        #    # similarities.append(float(b['similarity'][0]))
        #    sim_temp1 = [float(b) for b in batch["similarity"]]
        #    # similarities = similarities  +  [b for b in batch]
        #    similarities1 = similarities1 + sim_temp1

        #    sim_temp2 = [float(b) for b in batch["similarity2"]]
        #    similarities2 = similarities2 + sim_temp2
        ed = [[float(b) for b in batch["ed"]] for batch in dataloader]
        mces = [[float(b) for b in batch["mces"]] for batch in dataloader]

        ed = [item for sublist in ed for item in sublist]
        mces = [item for sublist in mces for item in sublist]
        return ed, mces
