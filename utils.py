
import re


def tokenize_with_mask(text, tokenizer, label=' '):
    """tokenize text while also generating a token-level mask based on <label></label> tags in the text. 
       tokens enclosed in <label></label> tags will have mask value 1, while others tokens will have mask value 0.
    """
    # find and remove all other tags in the form of <...> </...> except <label> </label>
    text = re.sub(rf'</?(?!{label}\b)\w+[^>]*>', '', text)

    # find all spans of text enclosed in tags, and calculate the span in the original text without tags
    tag_left, tag_right = f'<{label}>', f'</{label}>'
    spans = []
    cur_pos = 0
    while True:
        start_pos = text.find(tag_left, cur_pos)
        if start_pos == -1:
            break
        else:
            end_pos = text.find(tag_right, start_pos)
            span_start = start_pos - (len(tag_left) + len(tag_right)) * len(spans)
            if start_pos > 0 and text[start_pos - 1] == ' ':
                span_start -= 1           # span includes the space before the first word, as token_to_chars returned span includes the space before a word
            span_end = end_pos - len(tag_left) - (len(tag_left) + len(tag_right)) * len(spans)
            spans.append((span_start, span_end))         
            cur_pos = end_pos + 1
    
    # tokenize text and generate mask
    text = text.replace(tag_left, "").replace(tag_right, "")
    result = tokenizer(text, return_offsets_mapping=True, return_tensors=None)
    mask = []
    for token_start, token_end in result.offset_mapping:
        if any([token_start >= s[0] and token_end <= s[1] and token_end != token_start for s in spans]):    # only consider tokens strictly within a span
            mask.append(1)
        else:
            mask.append(0)
    
    return result, mask, text