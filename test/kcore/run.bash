declare -arr graph=(facebook europe.osm kron24 livejournal orkut pokec random roadNet-CA rmat uk2002 twitter)

path=/mnt/raid0_huge/hang/discovery_dataset
for file in ${graph[@]};
do
echo $file
echo ./kcore.bin $path/"$file"/"$file"_beg_pos.bin $path/"$file"/"$file"_csr.bin $path/"$file"/"$file"_weight.bin 1
./kcore.bin $path/"$file"/"$file"_beg_pos.bin $path/"$file"/"$file"_csr.bin $path/"$file"/"$file"_weight.bin 1
done

