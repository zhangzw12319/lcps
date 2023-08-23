# More Visualizations

![](https://s2.loli.net/2023/08/19/TV39IbcEpK1oA2n.png)
<div align="justify">
The overview of qualitative results from NuScenes validation set. (a) and (b) are visualization comparisons among the groundtruth (denoted as GT), the baseline predictions (Baseline), and full LCPS predictions (Full). Red circles emphasize the notable differences. We find that various Thing and Stuff objects can be predicted more accurately. (c) and (d) demonstrate semantic segmentation quality at nighttime. (e) and (f) verify the robust instance segmentation ability of our network. 
</div>

---



<img src="https://s2.loli.net/2023/08/19/YEhuRzaeyUltXvC.png" style="zoom: 100%;" />
<div align="justify">
Visual comparisons of asynchronous compensation. The first and the second lines are visualizations without or with asynchronous compensation respectively. The leftmost three columns demonstrate the effectiveness, especially for foreground objects of various sizes and geometric shapes. The last column specifies that the asynchronous compensation almost makes no difference when the time gap is small or at the front view. 
</div>



---

<img src="https://s2.loli.net/2023/08/19/J3ODqSXkGRhoexI.png" style="zoom:67%;" />
<div align="center">
The visualization results of semantic-aware regions filtered by CAMs.
</div>


---



![](https://s2.loli.net/2023/08/19/sMVqBGmHxXw59OU.png)
<div align="center">
Visualization results of SemanticKITTI validation set.
</div>

---

<img src="https://s2.loli.net/2023/08/24/TVZfhlX6QFejBu4.png" style="zoom:80%;" />

<div align="justify">
The visualization of correction ability between ground-truth (GT, first line) and our model predictions (P, second line). The results are from the NuScenes validation set, where the scan number below represents the sample index and red circles highlight notable differences. (a) shows that our model can segment an intact segmentation of truck, even though the ground-truth labels overlook the top area. (b) demonstrates that our model can consistently segment an entire vehicle during a sequence of frames over time, while the groundtruth labels miss the back surface when the ego-car advances at a high speed. 
</div>

---

<img src="https://s2.loli.net/2023/08/19/HKrpjUsc9LGf6Iz.png" style="zoom:33%;" />

<div align="justify">
Visualization results of foreground objects. GT represents ground-truth labels, while P represents predictions of our LCPS. Text such as ”Back” and ”FRONT LEFT” refers to the specific camera sensor. In this figure, the perspective projections of object segmentation are within the same image. Generally, our network achieves accurate segmentation results over these small and distant objects, such as bicycle and motorcycle, or rare objects like construction vehicle. 
</div>

---



![](https://s2.loli.net/2023/08/19/LHNdsikFKDUtEy7.png)

<div align="justify">
Visualization results of foreground objects. GT represents ground-truth labels, while P represents predictions of our LCPS. Texts like ”Back” and ”FRONT LEFT” refer to the specific camera sensor. This figure shows that most objects of diverse types and spatial locations across images can be consistently identified. 
</div>
