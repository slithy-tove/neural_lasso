rsync -av -e "ssh -S ~/.ssh/control-%r@%h:%p" \
  el862@bouchet.ycrc.yale.edu:/home/el862/project_pi_sk2433/el862/SNAB/results/ ./results/
