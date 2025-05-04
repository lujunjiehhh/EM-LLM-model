import sys
import os

# Add the current directory to the path to import the local Qwen3 model
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_qwen3_integration():
    print("Checking Qwen3 integration with EM-LLM...")
    
    # Check the modifications in patch_hf.py
    with open('em_llm/utils/patch_hf.py', 'r') as f:
        patch_hf_content = f.read()
    
    # Check if Qwen3ForCausalLM is included in the model check
    if "Qwen3ForCausalLM" in patch_hf_content:
        print("✅ Qwen3ForCausalLM is included in the model check")
    else:
        print("❌ Qwen3ForCausalLM is NOT included in the model check")
    
    # Check if the Qwen3-specific normalization handling is included
    if "q_norm" in patch_hf_content and "k_norm" in patch_hf_content:
        print("✅ Qwen3-specific normalization handling is included")
    else:
        print("❌ Qwen3-specific normalization handling is NOT included")
    
    # Check if the huggingface_forward function has been updated to handle extra kwargs
    if "**kwargs" in patch_hf_content:
        print("✅ huggingface_forward function has been updated to handle extra kwargs")
    else:
        print("❌ huggingface_forward function has NOT been updated to handle extra kwargs")
    
    # Check if em_llm_attn_forward has been updated
    with open('em_llm/attention/em_llm.py', 'r') as f:
        em_llm_content = f.read()
    
    if "extra_kwargs" in em_llm_content:
        print("✅ em_llm_attn_forward has been updated to handle extra kwargs")
    else:
        print("❌ em_llm_attn_forward has NOT been updated to handle extra kwargs")
    
    # Check if Qwen3-specific normalization is handled in em_llm.py
    if "q_norm = extra_kwargs.get('q_norm', None)" in em_llm_content:
        print("✅ Qwen3-specific normalization is handled in em_llm.py")
    else:
        print("❌ Qwen3-specific normalization is NOT handled in em_llm.py")
    
    # Overall assessment
    if all([
        "Qwen3ForCausalLM" in patch_hf_content,
        "q_norm" in patch_hf_content and "k_norm" in patch_hf_content,
        "**kwargs" in patch_hf_content,
        "extra_kwargs" in em_llm_content,
        "q_norm = extra_kwargs.get('q_norm', None)" in em_llm_content
    ]):
        return True
    else:
        return False

if __name__ == "__main__":
    success = test_qwen3_integration()
    if success:
        print("\n✅ Qwen3 integration with EM-LLM was successful!")
    else:
        print("\n❌ Qwen3 integration with EM-LLM failed.")