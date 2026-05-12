import pandas as pd
import re
import multiprocessing
import time
from pythainlp import sent_tokenize, word_tokenize
from pythainlp.util import normalize

def process_transcript_row(args):
    row_idx, text = args
    print(f"--- [Working] Processing Row {row_idx} ---")
    
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        print(f"xxx [Finished] Row {row_idx} (Empty or Invalid) xxx")
        return []

    results = []
    
    # 1. คลีนข้อความ ลบอักขระพิเศษ
    clean_text = re.sub(r'[^ก-๙a-zA-Z0-9\s]', ' ', text)
    clean_text = normalize(clean_text)
    
    # 2. ตัดประโยคด้วย crfcut (จะไม่หั่นตัวเลข/ภาษาอังกฤษมั่วแล้ว)
    # ต้องมั่นใจว่ารัน pip install python-crfsuite แล้ว
    sentences = sent_tokenize(clean_text, engine="crfcut")
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        # 3. ตัดคำด้วย newmm แล้วเชื่อมด้วยช่องว่าง
        words = word_tokenize(sent, engine="newmm")
        
        # กรองคำที่เป็นช่องว่างเปล่าๆ ออกก่อน join เพื่อไม่ให้เกิด space ซ้อนกันเยอะๆ
        words = [w.strip() for w in words if w.strip()]
        final_message = ' '.join(words)
        
        if final_message:
            results.append({
                'message': final_message,
                'class': '9arm'
            })
    
    print(f"+++ [Finished] Row {row_idx} Done! +++")
    return results

if __name__ == '__main__':
    start_time = time.time()
    
    print("Loading 'transcript_raw.csv'...")
    try:
        df = pd.read_csv('transcript_raw.csv')
    except FileNotFoundError:
        print("Error: ไม่พบไฟล์ 'transcript_raw.csv'")
        exit()

    text_col = 'message' if 'message' in df.columns else df.columns[0] 

    tasks = [(i, msg) for i, msg in enumerate(df[text_col])]
    
    num_cores = multiprocessing.cpu_count()
    print(f"Starting multiprocessing with {num_cores} cores for {len(tasks)} rows...\n")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_results_nested = pool.map(process_transcript_row, tasks)
    
    final_list = [item for sublist in all_results_nested for item in sublist]
    
    df_final = pd.DataFrame(final_list)
    df_final.to_csv("transcript.csv", index=False, encoding="utf-8-sig")
    
    end_time = time.time()
    
    print("-" * 30)
    print(f"All Tasks Completed in {end_time - start_time:.2f} seconds!")
    print(f"Total sentences extracted: {len(df_final)} rows")
    print("Saved successfully to 'transcript.csv'")