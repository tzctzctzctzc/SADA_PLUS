POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."

def marked_prompt(doc):
    return doc["question"] + "\n" + "\n".join(
        f"{chr(65+i)}. {choice}" for i, choice in enumerate(doc["options"])
    ) + POST_PROMPT

def content_for(doc, protocol, max_pixels=1605632):
    text = marked_prompt(doc)
    images = doc["image_list"]
    parts = text.split("<image>")
    assert len(parts)-1 == len(images), (doc["idx"], len(parts)-1, len(images))
    visual = [{"type":"image", "image":im, "min_pixels":256*28*28,
               "max_pixels":max_pixels} for im in images]
    if protocol == "explicit_indices":
        named = parts[0] + "".join(f"Image {i+1}"+s for i,s in enumerate(parts[1:]))
        return visual + [{"type":"text", "text":named}]
    assert protocol == "interleaved"
    content = [{"type":"text", "text":parts[0]}]
    for v, s in zip(visual, parts[1:]):
        content.extend([v, {"type":"text", "text":s}])
    return content
