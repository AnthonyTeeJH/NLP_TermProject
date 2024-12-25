import kagglehub
import pandas as pd

path = kagglehub.dataset_download("kishanvavdara/llm-prompt-recovery-mean-prompts")

print("Path to dataset files:", path)

test_df = pd.read_csv('/kaggle/input/llm-prompt-recovery/test.csv')
mean_prompts_df = pd.read_csv('/kaggle/input/llm-prompt-recovery-mean-prompts/mean_prompts.csv')

random_prompts = mean_prompts_df['rewrite_prompt'][0]

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'rewrite_prompt': test_df['original_text'].apply(
        lambda text: mean_prompts_df['rewrite_prompt'].iloc[0] + 
                     ' Plucrarealucrarealucrarealucrarea ' + 
                     text + 
                     ' Plucrarealucrarealucrarealucrarea'
    )
})

submission_df.to_csv('submission.csv', index=False)
submission_df.head()