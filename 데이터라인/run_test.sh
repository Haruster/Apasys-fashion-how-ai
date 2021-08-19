CUDA_VISIBLE_DEVICES="0" python3 ./main.py --mode test \
                                   --in_file_tst_dialog ./data/ac_eval_t1.wst.dev \
                                   --in_file_fashion ./data/mdata.wst.txt \
                                   --in_file_img_feats ./data/extracted_feat.json \
                                   --subWordEmb_path ./sstm_v0p5_deploy/sstm_v4p49_np_final_n36134_d128_r_eng_upper.dat \
                                   --model_path ./gAIa_model \
                                   --model_file gAIa-10.pt \
                                   --mem_size 16 \
                                   --key_size 300 \
                                   --hops 3 \
                                   --eval_node [6000,6000,6000,200][2000,2000] \
                                   --batch_size 100 \
                                   --use_multimodal True \


