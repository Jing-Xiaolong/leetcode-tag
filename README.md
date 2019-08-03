[h2pl/leetcode](https://github.com/h2pl/leetcode)的C++实现，附带一些扩充。用于秋招第一遍分tag刷题，查漏补缺，并建立手撸算法的基本手感。



- [算法思想](#算法思想)
  - [二分查找](二分查找)
  - [贪心思想](#贪心思想)
  - [双指针思想](#双指针思想)
  - 



# 算法思想

## 二分查找

[leecode.69 x的平方根 easy](https://leetcode-cn.com/problems/sqrtx/)

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

[leetcode.367 有效的完全平方数 easy](https://leetcode-cn.com/problems/valid-perfect-square/)

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

[leetcode.441 排列硬币 easy](https://leetcode-cn.com/problems/arranging-coins/)

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

[leetcode.50 Pow(x,n) middle](https://leetcode-cn.com/problems/powx-n/)

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

[剑指offer 数字在排序数组中出现的次数](https://www.nowcoder.com/practice/70610bf967994b22bb1c26f9ae901fa2?tpId=13&tqId=11190&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

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

## 贪心思想

[leetcode.445 分发饼干 easy](https://leetcode-cn.com/problems/assign-cookies/)

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

[leetcode.452 用最少数量的箭引爆气球 middle](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

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

[leetcode.135 分发糖果 hard](https://leetcode-cn.com/problems/candy/)

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

[leetcode.122 买卖股票的最佳时机 II easy](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

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

[leetcode.605 种花问题 easy](https://leetcode-cn.com/problems/can-place-flowers/)

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

[leetcode.665 非递减数列 easy](https://leetcode-cn.com/problems/non-decreasing-array/)

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

[leetcode.392 判断子序列 middle](https://leetcode-cn.com/problems/is-subsequence/)

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

[leetcode.763 划分字母区间 middle](https://leetcode-cn.com/problems/partition-labels/)

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

[leetcode.406 根据身高重建队列 middle](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

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

[leetcode.621 任务调度器 midle](https://leetcode-cn.com/problems/task-scheduler/)

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

[leetcode.861 翻转矩阵后的得分 middle](https://leetcode-cn.com/problems/score-after-flipping-matrix/)

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

## 双指针思想

[leetcode.167 两数之和II - 输入有序数组 easy](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)

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

[leetcode.345 翻转字符串中的元音字母 easy](https://leetcode-cn.com/problems/reverse-vowels-of-a-string/)

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

[leetcode.633 平方数之和 easy](https://leetcode-cn.com/problems/sum-of-square-numbers/)

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

[leetcode.680 验证回文字符串 II easy](https://leetcode-cn.com/problems/valid-palindrome-ii/)

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

[leetcode.88 合并两个有序数组 easy](https://leetcode-cn.com/problems/merge-sorted-array/)

```c++
// 思路：从num1的m+n-1位置处往前填充数字
// nums1 = 1 2 3 0 0 0, nums2 = 2 3 5
//             p1   idx             p2
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

[leetcode.141 环形链表 easy](https://leetcode-cn.com/problems/linked-list-cycle/)

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

[剑指Offer - 链表中环的入口节点](https://www.nowcoder.com/practice/253d2c59ec3e4bc68da16833f79a38e4?tpId=13&tqId=11208&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

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

[leetcode.524 通过删除字母匹配到字典里最长单词 middle](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

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

