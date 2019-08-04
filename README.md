[h2pl/leetcode](https://github.com/h2pl/leetcode)的C++实现，附带一些扩充。用于秋招第一遍分tag刷题，查漏补缺，并建立手撸算法的基本手感。



- [算法思想](#算法思想)
  - [排序](#排序算法)
    - [七大基于比较的排序算法](#七大基于比较的排序算法)
    - [利用排序思想的算法](#利用排序思想的算法)
  - [二分查找](#二分查找)
  - [贪心思想](#贪心思想)
  - [双指针思想](#双指针思想)
  - [搜索](#搜索)
    - [广度优先搜索BFS](#广度优先搜索BFS)
    - [深度优先搜索DFS](#深度优先搜索)
- [数据结构](#数据结构)
  - [二叉树](#二叉树)
    - [二叉树的遍历](#二叉树的遍历)

<br>

# 算法思想

## 排序

### 七大基于比较的排序算法

#### 冒泡排序

```c++
// O(n^2)，稳定，从后往前遍历时发现逆序即立刻交换
void bubbleSort(vector<int> &ivec){
    int len = ivec.size();
    for(int i = 0; i < len; ++i)
        for(int j = 0; j < len - i - 1; ++j)
            if(ivec[j] > ivec[j + 1])	
                swap(ivec[j], ivec[j + 1]);
}
```

<br>

#### 选择排序

```c++
// O(n^2)，不稳定，第i次遍历时，选择下标[i, n)中数值最小的数放在i处
void selectionSort(vector<int> &ivec){
    int len = ivec.size();
    for(int i = 0; i < len; ++i){
        int minIdx = i;
        for(int j = i + 1; j < len; ++j)
            if(ivec[j] < ivec[i])	
                minIdx = j;
        swap(ivec[i], ivec[minIdx]);
    }
}
```

<br>

#### 插入排序

```c++
// O(n^2)，稳定，遍历第i个数时，一直将它往前交换直到不再比前一个数更大
void insertionSort(vector<int> &ivec){
	for(int i = 1; i < ivec.size(); ++i){
		int j = i - 1;
		while(j >= 0 && ivec[j + 1] < ivec[j])
			swap(ivec[j + 1], ivec[j]),	--j;
	}
}
```

<br>

#### 归并排序

```c++
// O(nlogn)，稳定，不断分成更小的数组进行归并
void merge(vector<int> &ivec, int low, int mid, int high){
    vector<int> helper;
    int L = low, R = mid + 1;
    while(L <= mid && R <= high)
        ivec[L] < ivec[R] ? helper.push_back(ivec[L++]) : helper.push_back(ivec[R++]);
    while(L <= mid)
        helper.push_back(ivec[L++]);
    while(R <= high)
        helper.push_back(ivec[R++]);
    for(int i = low; i <= high; ++i)
        ivec[i] = helper[i - low];
}
void mergeSort(vector<int> &ivec, int low, int high){
    if(low >= high)
        return;
    int mid = (high - low) / 2 + low;
    mergeSort(ivec, low, mid);
    mergeSort(ivec, mid + 1, high);
    merge(ivec, low, mid, high);
}
// 排序入口
void mergeSort(vector<int> &ivec){
    mergeSort(ivec, 0, ivec.size() - 1);
}
```

<br>

#### 快速排序

```c++
//O(nlogn)，不稳定，找到第i个数，使[0,i)中均比它小且(i,n)中均比它大，再对[0,i)和(i,n)进行快排
void quickSort(vector<int> &ivec, int low, int high){
	if(low >= high)
		return;
    // 找到pivot_value满足上述条件
	int pivot_value = ivec[low];
	int pivot = low;
	for(int i = low + 1; i <= high; ++i)
		if(ivec[i] < pivot_value){
			++pivot;
			swap(ivec[i], ivec[pivot]);
		}
	swap(ivec[low], ivec[pivot]);
	// 对[i,pivot),(pivot, n)继续快排
	quickSort(ivec, low, pivot - 1);
	quickSort(ivec, pivot + 1, high);
}
// 排序入口
void quickSort(vector<int> &ivec){
	quickSort(ivec, 0, ivec.size() - 1);
}
```

<br>

#### 堆排序

```c++
// 自己网上找资料看吧
void heapDown(vector<int> &ivec, int beg, int end){	// 调整 ivec[beg...end] 使之满足大顶堆要求
	for(int cur = beg, child = beg * 2 + 1; child <= end; cur = child, child = child * 2 + 1){
		if(child < end && ivec[child] < ivec[child + 1])
			++child;
		if(ivec[cur] > ivec[child])	// 调整父节点和子节点的关系
			break;
		else
			swap(ivec[cur], ivec[child]);
	}
}
// 排序入口
void heapSort(vector<int> &ivec){
	// 构造大顶堆
	for(int i = ivec.size() / 2 - 1; i >= 0; --i)
		heapDown(ivec, i, ivec.size() - 1);
	// 交换堆顶元素，并再次调整为大顶堆
	for(int i = ivec.size() - 1; i > 0; --i){
		swap(ivec[0], ivec[i]);
		heapDown(ivec, 0, i - 1);
	}
}
```

<br>

### 利用排序思想的算法

[leetcode.215 数组中的第K个最大元素 middle](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)（或前K个最大的元素）

> 在未排序的数组中找到第 **k** 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
>
> **示例:**
>
> ```
> 输入: [3,2,1,5,6,4] 和 k = 2
> 输出: 5
> 
> 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
> 输出: 4
> ```

```c++
// 思路：利用堆排序，第K次弹出大顶堆的堆顶即为第K大的元素。其中heapDown函数见堆排序
int findKthLargest(vector<int>& nums, int k) {
    // 构造大顶堆
    for(int i = nums.size() / 2 - 1; i >= 0; --i)
        heapDown(nums, i, nums.size() - 1);
    // 前k-1次换到数组末尾后，堆顶nums[0]即为第k大
    int cnt = 0;
    while(++cnt < k){	// 注意：条件 (++cnt < k) 进行 k-1 次操作， (cnt++ < k) 进行 k 次操作
        swap(nums[0], nums[nums.size() - cnt]);
        heapDown(nums, 0, nums.size() - cnt - 1);
    }
    return nums[0]; 	// k-1次操作之后的堆顶即为第k大的元素
}
```

<br>[剑指offer 数组中的逆序对](https://www.nowcoder.com/practice/96bd6684e04a44eb80e6a68efc0ec6c5?tpId=13&tqId=11188&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
>
> 题目保证输入的数组中没有的相同的数字, 数据范围：对于%50的数据,size<=10^4, 对于%75的数据,size<=10^5, 对于%100的数据,size<=2*10^5
>
> **示例：**
>
> ```
> 输入：1,2,3,4,5,6,7,0
> 输出：7
> ```

```c++
// 思路：在归并排序的过程中对逆序对进行统计
int InversePairs(vector<int> nums) {
    int cnt = 0;
    mergeCount(nums, 0, nums.size() - 1, cnt);
    return cnt;
}

void mergeCount(vector<int> &nums, int left, int right, int &cnt){
    if(left >= right)
        return;
    int mid = (right - left) / 2 + left;
    mergeCount(nums, left, mid, cnt);
    mergeCount(nums, mid + 1, right, cnt);
    merge(nums, left, mid, right, cnt);
}
void merge(vector<int> &nums, int left, int mid, int right, int &cnt){
    vector<int> tmp;
    int l = left, r = mid + 1;
    while(l <= mid && r <= right)
        if(nums[l] <= nums[r])
            tmp.push_back(nums[l++]);
    else{
        cnt = (cnt + mid - l + 1) % 1000000007;
        tmp.push_back(nums[r++]);
    }
    while(l <= mid)
        tmp.push_back(nums[l++]);
    while(r <= right)
        tmp.push_back(nums[r++]);
    for(int i = left; i <= right; ++i)
        nums[i] = tmp[i - left];
}
```

<br>

## 二分查找

[leecode.69 x的平方根 easy](https://leetcode-cn.com/problems/sqrtx/)

> 描述：实现 int sqrt(int x) 函数。计算并返回 x 的平方根，其中 x 是非负整数。由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
>
> **示例 :**
>
> ```
> 输入: 4
> 输出: 2
> 
> 输入: 8
> 输出: 2
> 说明: 8 的平方根是 2.82842..., 
>      由于返回类型是整数，小数部分将被舍去。
> ```

```C++
int mySqrt(int x) {
    if(x <= 1)
        return x;
    int low = 1, high = x;
    while(low <= high){
        int mid = (high - low) / 2 + low;
        if(x / mid == mid)
            return mid;
        else if(x / mid < mid)
            high = mid - 1;
        else
            low = mid + 1;
    }
    return high;
}
```

<br>[leetcode.367 有效的完全平方数 easy](https://leetcode-cn.com/problems/valid-perfect-square/)

> 给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。
>
> 说明：不要使用任何内置的库函数，如  sqrt。
>
> **示例 ：**
>
> ```
> 输入：16
> 输出：True
> 
> 输入：14
> 输出：False
> ```

```C++
bool isPerfectSquare(int num) {
    if(num < 1)
        return false;
    int low = 1, high = num;
    while(low <= high){
        int mid = (high - low) / 2 + low;
        if(num / mid == mid && num % mid == 0)
            return true;
        else if(num / mid < mid)
            high = mid - 1;
        else
            low = mid + 1;
    }
    return false;
}
```

<br>[leetcode.441 排列硬币 easy](https://leetcode-cn.com/problems/arranging-coins/)

> 你总共有 n 枚硬币，你需要将它们摆成一个阶梯形状，第 k 行就必须正好有 k 枚硬币。给定一个数字 n，找出可形成完整阶梯行的总行数。n 是一个非负整数，并且在32位有符号整型的范围内。
>
> **示例 :**
>
> ```
> n = 5
> 硬币可排列成以下几行:
> ¤
> ¤ ¤
> ¤ ¤
> 因为第三行不完整，所以返回2.
> ```

```C++
int arrangeCoins(int n) {
    if(n < 1)
        return 0;
    int low = 1, high = n;
    while(low <= high){
        int mid = (high - low) / 2 + low;
        long long total = (mid + 1) * (long long)mid / 2;
        if(total == n)
            return mid;
        else if(total < n)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return high;
}
```

<br>[leetcode.50 Pow(x,n) middle](https://leetcode-cn.com/problems/powx-n/)

> 实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数。
>
> **示例:**
>
> ```
> 输入: 2.00000, 10
> 输出: 1024.00000
> 
> 输入: 2.00000, -2
> 输出: 0.25000
> 解释: 2-2 = 1/22 = 1/4 = 0.25
> ```

```C++
// 思路：对x依次做1,2,4,...次幂，得到x, x*x, x*x*x*x，直到指数次数为不大于n的最大值
// 保存x*x*x*x...，并从n中减去该指数次数
// 重复操作，将所有x*x...连续乘起来即为所求
double myPow(double x, int n) {
    bool sign = n < 0;
    long long exp = abs((long long)n);  // 先转化为longlong再abs，防止出现n=INT_MIN
    double res = 1.0;
    while(exp > 0){
        long long e = 1;// 指数  1 -> 2   -> 4 ...
        double mul = x; // 结果  x -> x*x -> x*x*x*x ...
        while((e*2) < exp){   // 直到e*2仍小于exp
            e = e << 1;
            mul = mul * mul;
        }
        exp -= e;
        res *= mul;
    }
    return sign ? 1.0/res : res;
}
```

<br>[剑指offer 数字在排序数组中出现的次数](https://www.nowcoder.com/practice/70610bf967994b22bb1c26f9ae901fa2?tpId=13&tqId=11190&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

> 统计一个数字在排序数组中出现的次数。

```C++
// 思路：二分搜索num+0.5，num-0.5的位置，它们区间内个数即为所求
int biSearch(const vector<int> &data, double num){
    // 对数组data = 1 2 3a 3b 3c 4 4 5
    // num = 2.5 -> 返回2所在下标
    // num = 3.5 -> 返回3c所在下标
    int low = 0, high = data.size() - 1;
    while(low <= high){
        int mid = (high - low) / 2 + low;
        if(data[mid] < num)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return high;
}
int GetNumberOfK(vector<int> data ,int k) {
    return biSearch(data, k + 0.5) - biSearch(data, k - 0.5);
}
```

<br>[leetcode.162 寻找峰值 middle](https://leetcode-cn.com/problems/find-peak-element/)

> 峰值元素是指其值大于左右相邻值的元素。给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。你可以假设 nums[-1] = nums[n] = -∞。要求O(logN)时间复杂度
>
> **示例 1:**
>
> ```
> 输入: nums = [1,2,3,1]
> 输出: 2
> 解释: 3 是峰值元素，你的函数应该返回其索引 2。
> 
> 输入: nums = [1,2,1,3,5,6,4]
> 输出: 1 或 5 
> 解释: 你的函数可以返回索引 1，其峰值元素为 2；或者返回索引 5， 其峰值元素为 6。
> ```

```c++
// 思路：采用二分查找法，left = 0, right = nums.size() - 1为起点
// nums[mid] < nums[mid + 1]，局部上升，右侧必然有峰值       (注意，mid+1必然是存在的, mid-1不一定)
// nums[mid] > nums[mid + 1]，局部下降，左侧必然有峰值
int findPeakElement(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    while(left < right){
        int mid = (right - left) / 2 + left;
        if(nums[mid] < nums[mid + 1])	// 是否取等，后面if-else-都需要仔细调整
            left = mid + 1;
        else
            right = mid;
    }
    return right;        
}
```



<br>

## 贪心思想

[leetcode.445 分发饼干 easy](https://leetcode-cn.com/problems/assign-cookies/)

> 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。对每个孩子 i ，都有一个胃口值 gi ，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j ，都有一个尺寸 sj 。如果 sj >= gi ，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
>
> **注意：**
>
> 你可以假设胃口值为正。
> 一个小朋友最多只能拥有一块饼干。
>
> **示例 1:**
>
> ```
> 输入: [1,2,3], [1,1]
> 输出: 1
> 解释: 
> 你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
> 虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
> 所以你应该输出1。
> ```

```c++
// 思路：从胃口最小的孩子开始，从最小的饼干开始喂直到能满足为止
int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int cnt = 0;
    int i = 0, j = 0;
    // 从胃口最小的孩子喂饼干，从最小的饼干开始喂
    while(i < g.size() && j < s.size()){
        if(g[i] <= s[j]){
            ++cnt;
            ++i;
            ++j;
        }else
            ++j;
    }
    return cnt;
}
```

<br>[leetcode.452 用最少数量的箭引爆气球 middle](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

> 在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以y坐标并不重要，因此只要知道开始和结束的x坐标就足够了。开始坐标总是小于结束坐标。平面内最多存在104个气球。
>
> 一支弓箭可以沿着x轴从不同点完全垂直地射出。在坐标x处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
>
> **示例:**
>
> ```
> 输入:
> [[10,16], [2,8], [1,6], [7,12]]
> 输出:
> 2
> 解释:
> 对于该样例，我们可以在x = 6（射爆[2,8],[1,6]两个气球）和 x = 11（射爆另外两个气球）。
> ```

```c++
// 思路：左往右依次射箭，每次保证引爆最多的气球
int findMinArrowShots(vector<vector<int>>& points) {
    if(points.size() <= 1)
        return points.size();
    sort(points.begin(), points.end());
    int cnt = 1;
    int rightEdge = points[0][1];   // 第一个圆右侧为右边界
    for(int idx = 1; idx < points.size(); ++idx){
        // 圆在边界内，需更新边界为它们右侧的较小值
        if(points[idx][0] <= rightEdge)  
            rightEdge = min(rightEdge, points[idx][1]);
        // 圆不在边界内，引爆气球++cnt，并更新右边界为新的右侧值
        else{   
            ++cnt;
            rightEdge = points[idx][1];
        }
    }
    return cnt;
}
```

<br>[leetcode.135 分发糖果 hard](https://leetcode-cn.com/problems/candy/)

> 老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
>
> 你需要按照以下要求，帮助老师给这些孩子分发糖果：
>
> 每个孩子至少分配到 1 个糖果。
> 相邻的孩子中，评分高的孩子必须获得更多的糖果。
> 那么这样下来，老师至少需要准备多少颗糖果呢？
>
> **示例:**
>
> ```
> 输入: [1,0,2]
> 输出: 5
> 解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
> 
> 输入: [1,2,2]
> 输出: 4
> 解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。
>      第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
> ```

```c++
// 思路：
// 步骤1. 先每个人发一颗糖果
// 步骤2. 从左往右遍历，i比i-1表现好就在i-1基础上+1糖果
// 步骤3. 从右往左遍历，i比i+1表现好就在i+1基础上+1糖果（需判断已有糖果是否已经比i+1多）
int candy(vector<int>& ratings) {
    vector<int> candies(ratings.size(), 1);

    for(int i = 1; i < candies.size(); ++i)
        if(ratings[i] > ratings[i - 1])
            candies[i] = candies[i - 1] + 1;

    for(int i = candies.size() - 2; i >= 0; --i)
        if(ratings[i] > ratings[i + 1])
            candies[i] = max(candies[i], candies[i + 1] + 1);

    return accumulate(candies.begin(), candies.end(), 0);
}
```

<br>[leetcode.122 买卖股票的最佳时机 II easy](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

> 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）

```c++
// 思路：只要p[i] > p[i - 1]，就在i-1买入并在i卖出。某天既买入又卖出可视为没有做任何操作
int maxProfit(vector<int>& prices) {
    if(prices.size() <= 1)
        return 0;
    int prof = 0;
    for(int i = 1; i < prices.size(); ++i)
        prof += (prices[i] > prices[i - 1]) ? prices[i] - prices[i - 1] : 0;
    return prof;
}

```

<br>[leetcode.605 种花问题 easy](https://leetcode-cn.com/problems/can-place-flowers/)

> 给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 n 。能否在不打破种植规则的情况下种入 n 朵花？能则返回True，不能则返回False。

```c++
// 思路：直接遍历，满足条件就种下。注意两类特殊情况和花坛两端
bool canPlaceFlowers(vector<int>& bed, int n) {
    int seeds = 0;
    if(n == 0)
        return true;
    if(bed.size() == 1)
        return bed[0] == 0;

    if(bed.size() >= 2 && bed[0] == 0 && bed[1] == 0){
        ++seeds;
        bed[0] = 1;
    }
    for(int i = 1; i < bed.size() - 1; ++i)
        if(bed[i - 1] == 0 && bed[i] == 0 && bed[i + 1] == 0){
            ++seeds;
            bed[i] = 1;
        }
    if(bed.size() >= 2 && bed[bed.size() - 2] == 0 && bed.back() == 0)
        ++seeds;

    return seeds >= n;
}

```

<br>[leetcode.665 非递减数列 easy](https://leetcode-cn.com/problems/non-decreasing-array/)

> 给定一个长度为 n 的整数数组，你的任务是判断在最多改变 1 个元素的情况下，该数组能否变成一个非递减数列。我们是这样定义一个非递减数列的： 对于数组中所有的 i (1 <= i < n)，满足 array[i] <= array[i + 1]。
>
> **示例 1:**
>
> ```
> 输入: [4,2,3]
> 输出: True
> 解释: 你可以通过把第一个4变成1来使得它成为一个非递减数列。
> 
> 输入: [4,2,1]
> 输出: False
> 解释: 你不能在只改变一个元素的情况下将其变为非递减数列。
> ```

```c++
// 思路：对逆序的数对进行计数，同时需要确认是否可以修正
// 找到nums[i] > nums[i + 1]的位置，再对nums[i] or nums[i + 1]进行更改
bool checkPossibility(vector<int>& nums) {
    int cnt = 0;
    for(int i = 0; i < nums.size() - 1; ++i)
        if(nums[i] > nums[i + 1]){
            ++cnt;
            // 优先考虑更改nums[i]为它能接受的最小值，即nums[i - 1]
            if(i == 0)
                continue;
            if(nums[i - 1] <= nums[i + 1])
                nums[i] = nums[i - 1];	// 2 4(i) 2 5 => 改4为2
            else
                nums[i + 1] = nums[i];	// 3 4(i) 2 5 => 不能改4
        }
    return cnt <= 1;
}

```

<br>[leetcode.392 判断子序列 middle](https://leetcode-cn.com/problems/is-subsequence/)

> 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

```c++
// 思路：直接比对每一个字母即可
bool isSubsequence(string s, string t) {
    if(s.empty())
        return true;
    int p = 0;
    for(const auto &c:t)
        if(s[p] == c)
            if(++p == s.size())
                return true;
    return false;
}

```

<br>[leetcode.763 划分字母区间 middle](https://leetcode-cn.com/problems/partition-labels/)

> 字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一个字母只会出现在其中的一个片段。返回一个表示每个字符串片段的长度的列表。
>
> **示例 1:**
>
> ```
> 输入: S = "ababcbacadefegdehijhklij"
> 输出: [9,7,8]
> 解释:
> 划分结果为 "ababcbaca", "defegde", "hijhklij"。每个字母最多出现在一个片段中。像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少
> 
> ```

```c++
// 思路: 第一遍遍历得到每个字母最后出现的位置map
//       第二遍通过left,right指示区间，cur指示当前遍历到的位置
//       	使用right = max(map[s[cur]], right)扩张右侧边界
//       	cur==right时说明区间内的字母未出现在其它区域
vector<int> partitionLabels(string str) {
    map<char, int> map;
    for(int i = 0; i < str.size(); ++i)
        map[str[i]] = i;

    int left = 0, right = map[str[0]], cur = 0;
    vector<int> res;
    while(right < str.size()){
        right = max(right, map[str[cur]]);	// 更新右边界
        if(right == cur){				   // 此时[left,right]之间的字母未出现在其它位置
            res.push_back(right - left + 1);
            left = right + 1;
            right = (cur < str.size() - 1) ? map[str[cur + 1]] : str.size();
        }
        ++cur;
    }

    return res;
}

```

<br>[leetcode.56 合并区间 middle](https://leetcode-cn.com/problems/merge-intervals/)

> 给出一个区间的集合，请合并所有重叠的区间。
>
> 示例 1:
>
> ```
> 输入: [[1,3],[2,6],[8,10],[15,18]]
> 输出: [[1,6],[8,10],[15,18]]
> 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
> 
> 输入: [[1,4],[4,5]]
> 输出: [[1,5]]
> 解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
> ```

```c++
// 思路：
// 将所有区间从小到大排序
// 选择第一个区间右端点作为界限，不断向右扩展其势力范围内的其它区间
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if(intervals.size() <= 1)
        return intervals;
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> res;
    auto tmp = intervals[0];
    for(int i = 1; i < intervals.size(); ++i){
        if(intervals[i][0] <= tmp[1])
            tmp[1] = max(tmp[1], intervals[i][1]);
        else{
            res.push_back(tmp);
            tmp = intervals[i];
        }
    }
    res.push_back(it0);	// 最后一个tmp由于没有进入到for循环中的else语句，需要额外加入到res中
    return res;
}
```

<br>[leetcode.406 根据身高重建队列 middle](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

> 假设有打乱顺序的一群人站成一个队列。 每个人由一个整数对(h, k)表示，其中h是这个人的身高，k是排在这个人前面且身高大于或等于h的人数。 编写一个算法来重建这个队列。
>
> 注意：总人数少于1100人。
>
> 示例
>
> ```
> 输入:
> [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
> 输出:
> [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
> ```

```c++
// 思路
// 将所有人(h,k)按身高h降序，k升序进行排列得 [7,0],[7,1],[6,1],[5,0],[5,2],[4,4]
// 此时所有人前面的人均不低于它，只需将(hi,ki)往前swap直到ki = i为止
bool comp(const vector<int> &p1, const vector<int> &p2){
    return p1[0] > p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]);
}
vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
    sort(people.begin(), people.end(), comp);

    for(int i = 0; i < people.size(); ++i)
        for(int k = i; k > people[k][1]; --k)
            swap(people[k], people[k - 1]);

    return people;
}
```

<br>[leetcode.621 任务调度器 middle](https://leetcode-cn.com/problems/task-scheduler/)

> 给定一个用字符数组表示的 CPU 需要执行的任务列表。其中包含使用大写的 A - Z 字母表示的26 种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。CPU 在任何一个单位时间内都可以执行一个任务，或者在待命状态。然而，两个相同种类的任务之间必须有长度为 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。你需要计算完成所有任务所需要的最短时间。
>
> 示例 1：
>
> ```
> 输入: tasks = ["A","A","A","B","B","B"], n = 2
> 输出: 8
> 执行顺序: A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
> ```

```c++
// 思路：统计每个任务出现次数，假设为AAAABBBBCCC,n=3
// 放A: |A---|A---|A---|A 
// 放B：|AB--|AB--|AB--|AB   => 执行时间 = (4 - 1) * (n + 1) + 2
// 放C：|ABC-|ABC-|ABC-|AB	 其中4是最多的次数，2是最多次数的任务
// 另外，当出现AAABBBCCDD,n=2时，总能通过尽量稀疏分布使得不需要等待时间
// 放A：|A--|A--|A--|
// 放B: |AB-|AB-|AB-|
// 放C：|ABC|AB-|ABC|		(这里是关键)
// 放D：|ABC|ABD|ABC|D
int leastInterval(vector<char>& tasks, int n) {
    vector<int> stat(26, 0);
    for(const auto &c:tasks)
        ++stat[c - 'A'];
    sort(stat.begin(), stat.end());

    int max_task = stat[25], max_cnt = 0;
    for(const auto &cnt:stat)
        max_cnt += cnt == stat[25] ? 1 : 0;

    int time = (max_task - 1) * (n + 1) + max_cnt;
    time = time > tasks.size() ? time : tasks.size();

    return time;
}
```

<br>[leetcode.861 翻转矩阵后的得分 middle](https://leetcode-cn.com/problems/score-after-flipping-matrix/)

> 有一个二维矩阵 A 其中每个元素的值为 0 或 1 。移动是指选择任一行或列，并转换该行或列中的每一个值：将所有 0 都更改为 1，将所有 1 都更改为 0。在做出任意次数的移动后，将该矩阵的每一行都按照二进制数来解释，矩阵的得分就是这些数字的总和。返回尽可能高的分数。
>
> **示例：**
>
> ```
> 输入：[[0,0,1,1],[1,0,1,0],[1,1,0,0]]
> 输出：39
> 解释：转换为 [[1,1,1,1],[1,0,0,1],[1,1,1,1]] => 0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39
> ```

```c++
// 思路:首先每行翻转，保证第一位是1, 1000 > 0111
//      然后第二列开始，每列翻转保证该列1数量比0多
void reverseRow(vector<vector<int>> &A, int row){
    for(auto &c:A[row])
        c ^= 1;
}
void reverseCol(vector<vector<int>> &A, int col){
    for(int i = 0; i < A.size(); ++i)
        A[i][col] ^= 1;
}
int matrixScore(vector<vector<int>>& A) {
    if(A.empty() || A[0].empty())
        return 0;
    // 每行变换，保证每行第一位是1
    for(int row = 0; row < A.size(); ++row)
        if(A[row][0] == 0)
            reverseRow(A, row);
    // 列变换，保证每列1数量比0多
    for(int col = 1; col < A[0].size(); ++col){
        int ones = 0, zeros = 0;
        for(int i = 0; i < A.size(); ++i)
            A[i][col] == 0 ? ++ones : ++zeros;
        if(ones > zeros)
            reverseCol(A, col);
    }
    int sum = 0;	// 计算最终结果
    for(const auto &row:A){
        int sum_row = 0;
        for(const auto &i:row)
            sum_row = (sum_row << 1) + i;	// 移位运算符优先级很低，需要打上括号
        sum += sum_row;
    }
    return sum;
}
```

<br>

## 双指针思想

<br>[leetcode.167 两数之和II - 输入有序数组 easy](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)

```c++
// 思路：两个下标分别指向首、尾，同时向中间靠拢以搜索两数
vector<int> twoSum(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while(left <= right){
        int sum = nums[left] + nums[right];
        if(sum == target)
            return { left + 1, right + 1 };
        else if(sum < target)
            ++left;
        else
            --right;
    }
    return {0,0};	// 题目必有解，不会执行到这里，但没有这句话无法通过编译
}
```

<br>[leetcode.345 翻转字符串中的元音字母 easy](https://leetcode-cn.com/problems/reverse-vowels-of-a-string/)

```c++
const set<char> sc = { 'a','e','i','o','u' };
string reverseVowels(string s) {
    int left = 0, right = s.size() - 1;
    while(left <= right){
        while(left < s.size() && sc.count(tolower(s[left])) == 0)
            ++left;	// 左往右定位到元音
        while(right >= 0 && sc.count(tolower(s[right])) == 0)
            --right;// 右往左定位到元音
        if(left <= right)
            swap(s[left++], s[right--]);
    }
    return s;
}
```

<br>[leetcode.633 平方数之和 easy](https://leetcode-cn.com/problems/sum-of-square-numbers/)

```c++
bool judgeSquareSum(int c) {
    long long left = 0, right = sqrt(c);
    while(left <= right){
        long long num = left * left + right * right;
        if(num < c)
            ++left;
        else if(num > c)
            --right;
        else if(num == c)
            return true;                
    }
    return false;
}
```

<br>[leetcode.680 验证回文字符串 II easy](https://leetcode-cn.com/problems/valid-palindrome-ii/)

```c++
// 思路：定位到第一个非回文的位置，刨除左字母或右字母后验证剩余部分是否为回文串
bool valid(const string &s, int left, int right){
    while(left <= right)
        if(s[left++] != s[right--])
            return false;
    return true;
}
bool validPalindrome(string s) {
    int left = 0, right = s.size() - 1;
    while(left <= right){
        if(s[left] != s[right])	// 定位后验证
            return valid(s, left + 1, right) || valid(s, left, right - 1);
        ++left;
        --right;
    }
    return true;
}
```

<br>[leetcode.88 合并两个有序数组 easy](https://leetcode-cn.com/problems/merge-sorted-array/)

> 给定两个有序整数数组 *nums1* 和 *nums2*，将 *nums2* 合并到 *nums1* 中*，*使得 *num1* 成为一个有序数组。
>
> **示例:**
>
> ```
> 输入:
>     nums1 = [1,2,3,0,0,0], m = 3
>     nums2 = [2,5,6],       n = 3
> 输出: [1,2,2,3,5,6]
> ```

```c++
// 思路：从num1的m+n-1位置处往前填充数字
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int idx = m + n - 1;
    int p1 = m - 1, p2 = n - 1;
    while(idx >= 0)
        if(p1 >= 0 && p2 >= 0){ // 在num1,nums2中都还有未融合的数才进行
            nums1[idx--] = max(nums1[p1], nums2[p2]);
            nums1[p1] > nums2[p2] ? --p1 : --p2;
        }else
            break;
    // p2>=0说明nums2中存在数没融合，将它放到nums1中； p1>=0，此时nums1中已经排好无需再管
    while(p2 >= 0)
        nums1[idx--] = nums2[p2--];
}
```

<br>[leetcode.283 移动零 easy](https://leetcode-cn.com/problems/move-zeroes/)

> 给定一个数组 `nums`，编写一个函数将所有 `0`移动到数组的末尾，同时保持非零元素的相对顺序。
>
> **示例:**
>
> ```
> 输入: [0,1,0,3,12]
> 输出: [1,3,12,0,0]
> ```

```c++
// 与leetcode.88类似，不过这里从前往后填充
void moveZeroes(vector<int>& nums) {
    int p1 = 0, p2 = 0;
    while(p2 < nums.size()){
        if(nums[p2] != 0)
            nums[p1++] = nums[p2];
        ++p2;
    }
    while(p1 < nums.size())
        nums[p1++] = 0;
}

// 写得更紧凑一些
void moveZeros(vector<int> &nums){
    int p = 0;
    for(auto num:nums)
        if(num != 0)
            nums[p++] = num;
    while(p < nums.size())
        nums[p++] = 0;
}
```



<br>[leetcode.19 删除链表的倒数第N个节点 middle](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

```c++
// 思路：快慢指针，快指针先走n次，然后快、慢一起走，快指针到达nullptr时慢指针即倒数第n个
// 注意：题目保证n有效，即1 <= n <= 总节点数
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode *fast = head, *slow = head;
    int cnt = 0;
    while(cnt++ < n)
        fast = fast->next;
    // n == 总节点数，倒数第n个节点即头结点
    if(fast == nullptr){
        ListNode *tmp_head = head->next;
        delete head;
        return tmp_head;
    }
    // 其余情况：条件fast->next != nullptr可将slow定位到待删节点前一个
    //          条件fast != nullptr可将slow定位到待删节点。    注意区分使用
    while(fast->next != nullptr){
        fast = fast->next;
        slow = slow->next;
    }
    auto tmp = slow->next->next;
    delete slow->next;
    slow->next = tmp;
    return head;
}
```

<br>[leetcode.141 环形链表 easy](https://leetcode-cn.com/problems/linked-list-cycle/)

```c++
// 思路：快慢指针法，若有环则必然在某个点相遇
// 另外：也可以用set或者map对已出现的节点进行记录，需要额外空间O(n)
bool hasCycle(ListNode *head) {
    ListNode *fast = head, *slow = head;
    while(fast != nullptr && fast->next != nullptr){
        fast = fast->next->next;
        slow = slow->next;
        if(fast == slow)
            return true;
    }
    return false;
}
```

<br>[剑指Offer - 链表中环的入口节点](https://www.nowcoder.com/practice/253d2c59ec3e4bc68da16833f79a38e4?tpId=13&tqId=11208&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

```c++
// __a__.______ b  a,b,c分为为各段长度
//     c \_____*  <---- 快慢指针相遇点
//  若有环，快慢指针在图中 * 处相遇，存在关系 fast = a+b+(b+c)*k = 2*slow =2*(a+b), k为快指针圈数
//  解得： a = (b + c)*(k - 1) + c
//  即：从头结点和相遇点分别出发，必然在入口节点相遇
ListNode* EntryNodeOfLoop(ListNode* head){
    ListNode* fast = head, *slow = head;
    while(fast != nullptr && fast->next != nullptr){
        fast = fast->next->next;
        slow = slow->next;
        if(slow == fast){	// 在*相遇
            fast = head;	// fast从头出发，slow从相遇点出发，一次跳一格
            while(fast != slow){
                fast = fast->next;
                slow = slow->next;
            }
            return fast;	// 入口接单相遇后返回fast或slow都可以
        }
    }
    return nullptr;
}
```

<br>[leetcode.3 无重复字符的最长子串 middle](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

> 给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。
>
> **示例 1:**
>
> ```
> 输入: "abcabcbb"
> 输出: 3 
> 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
> 
> 输入: "bbbbb"
> 输出: 1
> 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
> 
> 输入: "pwwkew"
> 输出: 3
> 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
>      请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
> ```

```c++
// 思路：用left, right表示无重复字符的窗口，用一个map记录right之前所有字符最后出现的位置loc
// 依次遍历每个字符的位置作为右端right，判断s[right]最后出现的位置是否在[left, right]内
int lengthOfLongestSubstring(string s) {
    int left = 0, maxLen = 0;
    map<char, int> loc; // 记录已遍历字符最后出现的位置
    for(int right = 0; right < s.size(); ++right){
        if(loc.count(s[right]) > 0 && loc[s[right]] >= left)	// 判断
            left = loc[s[right]] + 1;
        loc[s[right]] = right;
        maxLen = max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

<br>[leetcode.524 通过删除字母匹配到字典里最长单词 middle](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

> 给定一个字符串和一个字符串字典，找到字典里面最长的字符串，该字符串可以通过删除给定字符串的某些字符来得到。如果答案不止一个，返回长度最长且字典顺序最小的字符串。如果答案不存在，则返回空字符串。
>
> **示例 1:**
>
> ```
> 输入:s = "abpcplea", d = ["ale","apple","monkey","plea"]
> 输出: "apple"
> 
> 输入:s = "abpcplea", d = ["a","b","c"]
> 输出: "a"
> ```

```c++
// 思路:为字典里每个字符串分配一个指针idx, 表示该字符串已经匹配到了idx
//      遍历给定字符串str每个字母c，看c是否能与字典中字符串target_s[idx]进行匹配，匹配则idx+1
//      遍历完成后，看字典中每个字符串的idx==target_s.size()与否，相等则表示已匹配到
string findLongestWord(string str, vector<string>& vs) {
    unordered_map<string, int> dict;
    for(const auto &item:vs)
        dict[item] = 0;		// 分配指针
    
    for(const auto &c:str) // 比对
        for(auto &m:dict)  // 注意，这里一定要 auto &，否则更改后不会作用到m上
            if(c == m.first[m.second])
                ++m.second;

    string res = "";
    int maxLen = 0;
    for(const auto &m:dict)
        if(m.second == m.first.size()){ // 匹配成功的
            if(m.first.size() > maxLen){
                res = m.first;
                maxLen = m.first.size();
            }else if(m.first.size() == maxLen)
                res = res < m.first ? res : m.first;    // 字典序更小的
        }

    return res;
}
```



## 搜索

广度优先搜索BFS、深度优先搜索DFS是最长用的两种搜索方法，广泛应用在图、二维数组、树的搜索和遍历中。它们最本质的区别：<font color="red">**BSF是先入先出的遍历过程，DFS是先入后出的遍历过程**</font>；因此，在搜索过程中，<font color="red">**BFS一般借助于队列，DFS一般借助于栈**</font>，这一点要非常明确！

### 广度优先搜索BFS

[leetcode.102 二叉树的层次遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

> 二叉树的层次遍历是及其典型的广度优先搜索。广度优先即：一层一层向下，距离根同一距离的节点遍历完后再遍历深一层的节点，例如：
>
> ```
> 给定二叉树: [3,9,20,null,null,15,7],
> 	3
>    / \
>   9  20
>     /  \
>    15   7
> 层次遍历结果为：[[3], [9,20], [15,7]]
> ```

```c++
// 注意：返回结果是每一层的节点作为一个vector，所有层再作为一个vector
vector<vector<int>> levelOrder(TreeNode* root) {
    if(root == nullptr)
        return {};
    vector<vector<int>> res;
    queue<TreeNode*> q;
    q.push(root);
    while(!q.empty()){
        int cnt = q.size(); // 每层的节点数量
        vector<int> level_nums;
        while(cnt-- > 0){	
            auto tmp = q.front();
            q.pop();
            level_nums.push_back(tmp->val);
            if(tmp->left != nullptr)
                q.push(tmp->left);
            if(tmp->right != nullptr)
                q.push(tmp->right);
        }
        res.push_back(level_nums);
    }
    return res;
}

// 如果整体返回一个 vector<int>，则写法如下：
vector<int> levelOrder(TreeNode* root) {
    if(root == nullptr)
        return {};
    vector<int> res;
    queue<TreeNode*> q;
    q.push(root);
    while(!q.empty()){
        auto tmp = q.front();
        q.pop();
        res.push_back(tmp->val);
        if(tmp->left != nullptr)
            q.push(tmp->left);
        if(tmp->right != nullptr)
            q.push(tmp->right);
    }
    return res;
}
```

<br>[leetcode.199 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

> 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
>
> 示例:
>
> ```
> 输入: [1,2,3,null,5,null,4]
> 输出: [1, 3, 4]
> 解释:
> 
>    1            <---
>  /   \
> 2     3         <---
>  \     \
>   5     4       <---
> ```

思路：使用层次遍历，每层遍历最后一个节点时，保存节点的值

注：使用深度优先搜索也可以求解，查看[数据结构-树-树的遍历](#二叉树的遍历)、或[深度优先搜索DFS](#深度优先搜索DFS)

<br>[leetcode.101 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

> 给定一个二叉树，检查它是否是镜像对称的。
>
> ```
>     1
>    / \
>   2   2
>  / \ / \
> 3  4 4  3
> ```

```c++
// 递归法，递归不需要借助queue
bool isSymmetric(TreeNode* root) {
    if(root == nullptr)
        return true;
    return check(root->left, root->right);
}
bool check(TreeNode* l, TreeNode *r){
    if(l == nullptr && r == nullptr)
        return true;
    if(l == nullptr || r == nullptr || l->val != r->val)
        return false;
    return check(l->left, r->right) && check(r->left, l->right);
}

// 迭代法，借助queue实现
bool isSymmetric(TreeNode* root) {
    if(root == nullptr)
        return true;
    queue<TreeNode*> ql;
    queue<TreeNode*> qr;
    ql.push(root->left);
    qr.push(root->right);
    while(!ql.empty() && !qr.empty()){
        auto l = ql.front();
        auto r = qr.front();
        ql.pop();
        qr.pop();
        if(l == nullptr && r == nullptr)
            continue;
        if(l == nullptr || r == nullptr || l->val != r->val)
            return false;
        ql.push(l->left);
        qr.push(r->right);
        ql.push(l->right);
        qr.push(r->left);
    }
    return true;
}
```

<br>[leetcode.127 单词接龙 middle](https://leetcode-cn.com/problems/word-ladder/)

> 给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：每次转换只能改变一个字母;转换过程中的中间单词必须是字典中的单词。说明:
>
> 如果不存在这样的转换序列，返回 0。
> 所有单词具有相同的长度。
> 所有单词只由小写字母组成。
> 字典中不存在重复的单词。
> 你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
>
> 示例 1:
>
> ```
> 输入:
> beginWord = "hit",
> endWord = "cog",
> wordList = ["hot","dot","dog","lot","log","cog"]
> 输出: 5
> 解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
>      返回它的长度 5。
> ```

```c++
// 对beginword里的每一个字母，将它替换为26个字母（26个搜索方向），若替换后存在于wordList中，则可以作为下一层
// 示例1的搜索变换过程为 [hit] -> [hot] -> [dot, lot] -> [dog, log] -> [cog]
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> dict(wordList.begin(), wordList.end());
    if(dict.count(endWord) == 0)   // 词典中必然存在endWord才能转换
        return 0;
    dict.erase(beginWord);

    queue<string> q;    // 一般使用queue辅助广度优先搜索
    q.push(beginWord);
    int level = 1;      // beginWord算一层

    while(!q.empty()){
        int node_num = q.size();	// 该层的节点数
        while(node_num-- > 0){		// 遍历该层的所有节点
            string node = q.front();
            q.pop();
            if(node == endWord)
                return level;
            for(int i = 0; i < node.size(); ++i){	// 对每个字母
                char node_i = node[i];
                for(char c = 'a'; c < 'z'; ++c){	// 26个搜索方向，即替换为a-z
                    node[i] = c;
                    if(dict.count(node) > 0){
                        q.push(node);
                        dict.erase(node);   // 剔除该节点，防止后面再次搜索到
                    }
                }
                node[i] = node_i;
            }
        }
        ++level;
    }
    return 0;   // 遍历到这说明没搜索到
}
```



### 深度优先搜索DFS

[leetcode.199 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

> 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
>
> 示例:
>
> ```
> 输入: [1,2,3,null,5,null,4]
> 输出: [1, 3, 4]
> 解释:
> 
>    1            <---
>  /   \
> 2     3         <---
>  \     \
>   5     4       <---
> ```

```c++
// 思路：DFS-先序遍历（右侧开始），某层最先达到的节点即为该层最右侧的节点
vector<int> rightSideView(TreeNode* root) {
    if(root == nullptr)
        return {};

    vector<int> res;
    preorder(root, 0, res);        
    return res;
}
void preorder(TreeNode *root, int level, vector<int> &nums){
    if(root == nullptr)
        return;
    if(nums.size() == level)
        nums.push_back(root->val);
    preorder(root->right, level + 1, nums);
    preorder(root->left, level + 1, nums);
}
// 此题也可使用BFS做，即层次遍历，每层最后一个节点放入vector中即可
// 注：树的前、中、后序遍历均是深度优先搜索
```













<br><br><br>

# 数据结构

## 二叉树

### 二叉树的遍历

[leetcode.199 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

> 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
>
> 示例:
>
> ```
> 输入: [1,2,3,null,5,null,4]
> 输出: [1, 3, 4]
> 解释:
> 
>    1            <---
>  /   \
> 2     3         <---
>  \     \
>   5     4       <---
> ```

```c++
// 思路：DFS-先序遍历（右侧开始），某层最先达到的节点即为该层最右侧的节点
vector<int> rightSideView(TreeNode* root) {
    if(root == nullptr)
        return {};

    vector<int> res;
    preorder(root, 0, res);        
    return res;
}
void preorder(TreeNode *root, int level, vector<int> &nums){
    if(root == nullptr)
        return;
    if(nums.size() == level)
        nums.push_back(root->val);
    preorder(root->right, level + 1, nums);
    preorder(root->left, level + 1, nums);
}

// 此题也可使用BFS做，即层次遍历，每层最后一个节点放入vector中即可
```

