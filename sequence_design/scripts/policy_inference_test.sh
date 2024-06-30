#### BELOW CODE IS USED TO TEST INFERENCE OF FINETUNED POLICY MODEL ####

# Start the server
CUDA_VISIBLE_DEVICES=2 API_PORT=1338 python -m llmtunner.api_demo \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ckpts/policy \
    --template llama3 \
    --vllm_gpu_util 0.9 \
    --temperature 0.6 \
    --top_k 50 \
    --top_p 1.0 \
    --infer_backend vllm \
    --vllm_enforce_eager

# Call the API
curl http://localhost:1338/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [
    {"role": "user", "content": "Edit 1 amino acid in the below protein sequence to create a new protein with higher fluorescence. The amino acid must be in set {{L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C, X, B, U, Z, O}}. Please return only the modified protein sequence.\nProtein sequence: SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTNFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEGGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYO\nModified protein sequence: "}
    ]
    }'

# A sample response
# Response: "After analyzing the protein sequence and its corresponding fluorescence value, I identified a potential amino acid that could be edited to increase the fluorescence.\n\nThe edited amino acid is E at position 14, which is replaced with F. This modification is within the set of allowed amino acids.\n\nHere is the modified protein sequence:\n\nSKGFLLFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTNFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEGGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYO\n\nPlease note that this is just a suggestion, and the actual outcome may vary depending on the specific protein and its biological context."


# After Five iterations ==> Worse than before???
# "I'd be happy to help!\n\nTo increase the fluorescence of the protein, I'll introduce a small modification to the sequence. After analyzing the protein sequence, I'll make a single substitution to increase the fluorescence value.\n\nThe modified protein sequence with increased fluorescence is:\n\nSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTNFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEGGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYO -> SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTNFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEGGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYO -> SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTNFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEGGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYO -> SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTNFFKDDGNYKTRAEVKFEGDTLVNRI"