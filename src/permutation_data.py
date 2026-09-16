import re

def factorial_samples(dataset, config):
    requested=set(config["ids"])
    indices=[i for i,idx in enumerate(dataset["id"]) if idx in requested]
    selected={d["id"]:d for d in dataset.select(indices)}
    assert len(selected)==len(requested)
    expanded=[]
    for base_id in config["ids"]:
        d=selected[base_id]
        assert len(d["images"])==len(d["options"])==4
        original_indices=[int(re.search(r"image\s*(\d+)",s,re.I).group(1))-1 for s in d["options"]]
        gold_letter=d["answer"].strip("() \n")
        gold_image=original_indices[ord(gold_letter)-65]
        for pi,image_order in enumerate(config["image_permutations"]):
            correct_position=image_order.index(gold_image)
            for oi,option_order in enumerate(config["option_permutations"]):
                row=d | {"id":f"{base_id}:i{pi}:o{oi}","base_id":base_id,
                    "images":[d["images"][i] for i in image_order],
                    "options":[f"({chr(65+j)}) Image {image+1}" for j,image in enumerate(option_order)],
                    "answer":chr(65+option_order.index(correct_position)),
                    "image_order":image_order,"option_order":option_order,"gold_original_image":gold_image}
                expanded.append(row)
    return expanded
