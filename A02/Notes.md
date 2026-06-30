# Notes  
Calculated L2 Norms  
```
backbone.conv1.weight 0.22009390592575073
backbone.bn1.weight 0.035532910376787186
backbone.bn1.bias 0.03265804052352905
backbone.layer1.0.conv1.weight 0.5384514927864075
backbone.layer1.0.bn1.weight 0.03159306198358536
backbone.layer1.0.bn1.bias 0.03323431685566902
backbone.layer1.0.conv2.weight 0.6535183191299438
backbone.layer1.0.bn2.weight 0.03810790553689003
backbone.layer1.0.bn2.bias 0.039614371955394745
backbone.layer1.1.conv1.weight 0.6105197072029114
backbone.layer1.1.bn1.weight 0.03562486544251442
backbone.layer1.1.bn1.bias 0.04325360059738159
backbone.layer1.1.conv2.weight 0.6273196339607239
backbone.layer1.1.bn2.weight 0.0285351499915123
backbone.layer1.1.bn2.bias 0.037891365587711334
backbone.layer2.0.conv1.weight 0.8202038407325745
backbone.layer2.0.bn1.weight 0.04476087540388107
backbone.layer2.0.bn1.bias 0.04231513291597366
backbone.layer2.0.conv2.weight 1.279126763343811
backbone.layer2.0.bn2.weight 0.04705311357975006
backbone.layer2.0.bn2.bias 0.04650809243321419
backbone.layer2.0.downsample.0.weight 0.28739112615585327
backbone.layer2.0.downsample.1.weight 0.044929083436727524
backbone.layer2.0.downsample.1.bias 0.04650809243321419
backbone.layer2.1.conv1.weight 1.20085608959198
backbone.layer2.1.bn1.weight 0.04109789803624153
backbone.layer2.1.bn1.bias 0.035752102732658386
backbone.layer2.1.conv2.weight 1.2461090087890625
backbone.layer2.1.bn2.weight 0.03824698179960251
backbone.layer2.1.bn2.bias 0.04471689462661743
backbone.layer3.0.conv1.weight 1.8035242557525635
backbone.layer3.0.bn1.weight 0.055178236216306686
backbone.layer3.0.bn1.bias 0.0505976565182209
backbone.layer3.0.conv2.weight 2.4884238243103027
backbone.layer3.0.bn2.weight 0.057585302740335464
backbone.layer3.0.bn2.bias 0.05243830755352974
backbone.layer3.0.downsample.0.weight 0.6050446629524231
backbone.layer3.0.downsample.1.weight 0.05307197943329811
backbone.layer3.0.downsample.1.bias 0.05243830755352974
backbone.layer3.1.conv1.weight 2.447890520095825
backbone.layer3.1.bn1.weight 0.0585145503282547
backbone.layer3.1.bn1.bias 0.05313987284898758
backbone.layer3.1.conv2.weight 2.4884791374206543
backbone.layer3.1.bn2.weight 0.059798043221235275
backbone.layer3.1.bn2.bias 0.05015558749437332
backbone.layer4.0.conv1.weight 3.6516470909118652
backbone.layer4.0.bn1.weight 0.07171551883220673
backbone.layer4.0.bn1.bias 0.07242155820131302
backbone.layer4.0.conv2.weight 5.0666890144348145
backbone.layer4.0.bn2.weight 0.06633604317903519
backbone.layer4.0.bn2.bias 0.07004110515117645
backbone.layer4.0.downsample.0.weight 1.272046685218811
backbone.layer4.0.downsample.1.weight 0.07830897718667984
backbone.layer4.0.downsample.1.bias 0.07004110515117645
backbone.layer4.1.conv1.weight 4.987730026245117
backbone.layer4.1.bn1.weight 0.07439848780632019
backbone.layer4.1.bn1.bias 0.0940907821059227
backbone.layer4.1.conv2.weight 5.216094493865967
backbone.layer4.1.bn2.weight 0.09941599518060684
backbone.layer4.1.bn2.bias 0.10390977561473846
backbone.fc.weight 0.3054368197917938
backbone.fc.bias 0.005473879165947437

```
  
From this we can clearly see that the highest changes (largest L2 norms) are observed in the later layers of the backbone, particularly the deeper layers This indicates that most of the learning during fine-tuning happens in the layers which are not catering to basic edge or corner detection but the ones which are detecting deeper features such as textures and everything . That is also the reason why we see a higher accuracy in the fine tuned model for certain datasets. The early layers remain relatively unchanged.   
This happens because early layers capture general visual features such as edges and textures, which are already useful from initialization and do not require significant updates. In contrast, later layers adapt more to the specific task and therefore show larger weight changes.  
This explains why the frozen backbone model can still achieve performance close to full fine tuning on small datasets, since most of the useful simpler features are already present in the frozen backbone and can already be identified by it to a great extent.   
