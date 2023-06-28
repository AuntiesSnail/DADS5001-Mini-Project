# **Mini-Project: Credit Card Customer Churning**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/5f1bb246-3930-4940-b427-9df929fbad5f)

## **Part 1 : Introduction**

Customer Churn Rate คืออัตราที่ลูกค้าหยุดใช้บริการในช่วงระยะเวลาใดเวลาหนึ่ง ซึ่งมีผลในการดำเนินธุรกิจที่ต้องการรักษาลูกค้าให้ใช้บริการกับเราให้ยาวนานและมากที่สุด ดังนั้น การป้องกันการยกเลิกใช้บริการของลูกค้าจึงมีความสำคัญมาก โดยการนำ Data ที่เป็นฐานลูกค้า มาจัดเป็น Customer Segment ด้วย RFM Model ร่วมกับข้อมูล Demographic เพื่อทำการวิเคราะห์และวางแผนการทำ Marketing กับลูกค้าในแต่ละกลุ่มให้ได้ตรงตามเป้าหมายและมีประสิทธิภาพมากที่สุด

ซึ่งชุดข้อมูลที่ใช้ในโปรเจคนี้ เป็นข้อมูลลูกค้า Consumer Credit Card Portfolio ของธนาคารแห่งหนึ่ง ที่ต้องการวิเคราะห์หาลูกค้าที่มีแนวโน้มจะยกเลิกการใช้บริการบัตรเครดิต เพื่อที่จะได้มอบการบริการที่ดีขึ้นและกระตุ้นการตัดสินใจของลูกค้าให้กลับมาใช้บริการอีกครั้ง


**1.1 ข้อมูล Dataset :**

>kaggle : Predicting Credit Card Customer Segmentation

>Credit: The original authors – Data Source

>License: CC0 1.0 Universal (CC0 1.0) - Public Domain Dedication

>No Copyright - You are allowed to copy, modify, distribute, and perform the work, even for commercial purposes, without the need for permission

**1.2 Hypothesis :**

การพิจารณา Customer Segment โดย RFM Model ร่วมกับข้อมูล Demographic จะช่วยทำให้เข้าใจพฤติกรรมของลูกค้าได้ดียิ่งขึ้น โดยเฉพาะกลุ่ม “At risk”, “Can’t lose them” และ “Hibernating” ที่มีแนวโน้มจะเลิกใช้บริการ ทำให้สามารถปรับกลยุทธ์ทางการตลาดและเสนอสิทธิพิเศษให้กับลูกค้าในแต่ละกลุ่มได้อย่างเหมาะสม


## **Part 2 : Exploration Data Analysis (EDA)**

**2.1 Review Data:**

ข้อมูลลูกค้าทั้งหมด 10,127 ราย ประกอบด้วย 21 ตัวแปร ได้แก่

|Variable|Definition|Variable Group|
|--------|--------|--------|
CLIENTNUM|รหัสลูกค้าธนาคาร||	
Attrition_Flag|	แสดงสถานะการใช้บริการของลูกค้า	
Customer_Age|อายุลูกค้า|Demographic
Gender|เพศลูกค้า|Demographic
Dependent_count|จำนวนผู้ที่อยู่ในความอุปการะ	
Education_Level|ระดับการศึกษา|	Demographic
Marital_Status|สถานภาพสมรส|	Demographic
Income_Category|	ช่วงรายได้ของลูกค้า|	Demographic
Card_Category|	ประเภทบัตรของลูกค้า	
Months_on_book	|จำนวนเดือนที่ลูกค้าใช้บริการ	
Total_Relationship_Count|	จำนวนผลิตภัณฑ์ที่ลูกค้าถือครองกับธนาคาร	|Demographic
Months_Inactive_12_mon|	จำนวนเดือนที่ลูกค้าไม่มีการเคลื่อนไหว ใน 12 เดือนล่าสุด	
Contacts_Count_12_mon	|จำนวนครั้งที่มีการติดต่อ ใน 12 เดือนล่าสุด	
Credit_Limit|	วงเงินบัตรเครดิต	
Total_Revolving_Bal|	ยอดค้างชำระ	
Avg_Open_To_Buy|	วงเงินเครดิตที่พร้อมใช้สำหรับการซื้อสินค้าหรือทำธุรกรรมใหม่ (คำนวณค่าเฉลี่ยของวงเงินเครดิตที่เปิดให้ใช้งานในระยะเวลา 12 เดือน)	
Total_Amt_Chng_Q4_Q1	|ยอดใช้จ่าย Q4 เทียบกับ Q1	
Total_Trans_Amt|	ยอดใช้จ่ายทั้งหมด ใน 12 เดือนล่าสุด	
Total_Trans_Ct	|จำนวนรายการทั้งหมด ใน 12 เดือนล่าสุด	
Total_Ct_Chng_Q4_Q1|	จำนวนรายการ Q4 เทียบกับ Q1	
Avg_Utilization_Ratio	|ยอดการใช้จ่ายต่อวงเงิน เป็นอัตราเท่าไหร่ กำหนดให้ 1 คือใช้จ่ายเต็มวงเงิน และ 0 คือไม่มีการใช้จ่าย คิดเป็นค่าเฉลี่ยจากทั้งหมด 	

**_Note:_**

>1. จากข้อมูลทั้งหมด 23 ตัวแปร ซึ่งไม่มีตัวแปร Null และซึ่งไม่ใช้ตัวแปร Naive_Bayes1, Naive_Bayes2
 
>2. มีจำนวนทั้งหมด 3 ตัวแปร ได้แก่ Education_Level, Marital_Status, Income_Category พบข้อมูล Unknown ที่ไม่บงบอกสถานะที่แท้จริง

**2.2 Categorical and Numerical Variable:**

ในข้อมูลตัวแปร Attrition_Flag มี Attrited Customer (ลูกค้าที่ยกเลิกบริการ) 1,627 ราย (16.07%) และ Existing Customer (ลูกค้าที่ยังใช้บริการ) 8,500 ราย (83.93%)

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/9c66b2c9-28bd-4901-9c7d-c897a12a40f2)

* **Categorical Data**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/61273a0d-f84a-43de-9dc2-f0e9658218b6)

>บางตัวแปรข้อมูลระบุ Unknown

>>Education_Level มีอยู่ 1,519 ราย เป็น Attrited Customer 256 ราย

>>Marital_Status มีอยู่ 749 ราย เป็น Attrited Customer 129 ราย

>>Income_Category มีอยู่ 1,112 ราย เป็น Attrited Customer 187 ราย

>Card_Category เป็นประเภท Blue เกือบทั้งหมดเลย มี 9,436 คน (93.18%)


* **Numerical Data**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/e471245b-e7cf-4db9-9da4-03acd6676aca)

>ข้อมูลค่อนข้าง clean แต่ประเด็นที่น่าสงสัยคือ Avg_Utilization_Ratio เท่ากับ 0 คือลูกค้าไม่เคยใช้บัตรเลย มีอยู่ 2,470 คน แต่ที่ตัวแปร Months_on_book พบว่าเป็นลูกค้าใช้บริการมาแล้ว 13 เดือนขึ้นไปทุกคน และใช้บริการมากสุด 56 เดือน (เกือบ 5 ปี) แต่ไม่เคยใช้บัตร

>ตัวแปร Attrition_Flag พบว่าลูกค้า ที่ Avg_Utilization_Ratio เท่ากับ 0 เป็น Attrited Customer 893 ราย ซึ่งมากกว่า 50.00 % ของ Attrited Customer ทั้งหมด

>ตัวแปรส่วนใหญ่ เบ้ขวา


**2.3 การวิเคราะห์ RFM Model:**

RFM Analysis เป็น Model พื้นฐานเพื่อแบ่งกลุ่มลูกค้าหรือ Customer Segmentation ตามพฤติกรรมการใช้จ่าย โดยตัวแปรที่นำมาวิเคราะห์มี 3 ตัวแปรหลัก คือ R F และ M

  - Recency   (R) คือระยะเวลาที่ลูกค้าใช้งานบัตร ซึ่งจะใช้ตัวแปร Months_on_book 
  - Frequency (F) คือความถี่ในการใช้จ่ายของลูกค้า ซึ่งจะใช้ตัวแปร Total_Trans_C
  - Monetary  (M) คือยอดธุรกรรมทั้งหมด ซึ่งจะใช้ตัวแปร Total_Trans_Amt 

**_ตารางแสดงข้อมูลตัวอย่าง R F M_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/c0f8c169-57fb-48d7-8918-bbe1bde050bd)

**_การกำหนดเกณฑ์ให้คะแนน_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/4a84e2f8-0e61-4318-a938-525380c5b808)

จากรูปเป็นการแบ่งตามหลักการ Quintile โดยจะกำหนดเกณฑ์คะแนนตั้งแต่ 1 ถึง 5 และแบ่งจำนวนลูกค้าออกเป็นทั้งหมด 5 ช่วง เพื่อให้คะแนนในแต่ละด้านของ RFM Model วิธีการแบ่งกลุ่มคือการทำ Binning จากผลลัพธ์ที่ได้จาก .groupby() ข้างต้น

**_ตารางแสดงข้อมูลตัวอย่าง Rank R F M_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/20228d97-d64b-42f6-b7c1-d3a3331d9106)

ผลลัพธ์ที่ได้คือตัวเลข 1 ถึง 5 ที่เรียงลำดับตามพฤติกรรมที่เกิดขึ้น เช่น ถ้ามีค่า Recency น้อย ๆ จะถูกจัดอยู่ใน Binning ที่สูง หมายความว่าจำนวนวันที่เกิด Transaction ล่าสุด “ไม่นาน” โดยยิ่งลูกค้ามีการใช้บริการครั้งล่าสุดเร็วเท่าไหร่ ก็จะง่ายต่อการติดต่อกลับเพื่อ Engage

ถ้าหากมีค่า Frequency หรือมีความถี่ในการทำธุรกรรมบ่อยๆ จะถูกจัดอยู่ใน Binning ที่สูง (ไปในทิศทางเดียวกัน)  ส่วนค่าของ Monetary หรือยอดธุรกรรมมักจะสอดคล้องและไปเป็นในทิศทางเดียวกับค่า Frequency 

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/afe44d47-921a-417f-aa44-78f1e5cbdf2e)

หลังจากนั้นจัดทำ Segment เพื่อครอบคลุมเงื่อนไข RFM ทั้งหมด 125 (5x5x5) รูปแบบ และแบ่ง Customer Segment  ได้ทั้งหมด 11 Segment ได้แก่

|Segment|Description|
|-------|-----------|
01.Champions |Bought recently, buy often and spend the most
02.Loyal Customers|	Spend good money. Responsive to promotions
03.Potential Loyalist|	Recent customers, spent good amount, bought more than once
04.Recent Customers	|Bought more recently, but not often
05.Promising		|Recent shoppers, but haven’t spent much
06.Need Attention|	Above average recency, frequency & monetary values
07.About to Sleep	|Below average recency, frequency & monetary values
08.At Risk		|Spent big money, purchased often but long time ago
09.Can’t Lose Them	|Made big purchases and often, but long time ago
10.Hibernating	|Low spenders, low frequency, purchased long time ago
11.Lost		|Lowest recency, frequency & monetary scores

**_Source:_** 
>https://www.putler.com/rfm-analysis/

**_กราฟแสดงความสัมพันธ์ระหว่าง Score Recency Frequency Monetary_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/765163af-f632-42d4-846b-f40f459d0b89)


## **Part 3 : Insight**

**3.1 Visualization**

* **RFM analysis**

**_กราฟแสดง Customer Segment (Tree Map)_** 
![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/dc1c71e5-c343-4432-830a-a2ebb0cd1836)

**_กราฟเพื่อดูความสัมพันธ์ของ Recency Frequency Monetary ในกลุ่มลูกค้าแต่ละ Segment_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/d10587b6-7bda-4368-ba97-b0ca1b8cee9b)

**_กราฟแสดงค่าเฉลี่ยของยอดธุรกรรม_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/672fd73c-71b8-43e4-83d2-de5dd04c2d81)

**_กราฟแสดง Customer Segment (Scatter plot)_** 

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/f6bebab1-da77-4e08-bdd8-16332a32cf0d)

**_กราฟแสดงการเปรียบเทียบ Customer Segment 8,9 และ 10 กับ Segment อื่นๆ_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/2b5e308a-8116-4d99-a513-ab29a7f270ab)

* **Churning**

จากการทำ Model RFM พบว่า กลุ่มที่ใช้งานอยู่มีโอกาสเป็นกลุ่ม lost ถึง 6.7% (กลุ่มที่คาดหวังที่จะกลับมาใช้บริการบัตรน้อย) และกลุ่มที่เราสามารถนำเสนอเพื่อกระตุ้น และคาดหวังลูกค้ายังคงใช้บริการต่อไป ได้แก่ 

08.At risk	จำนวน 15.1%

09.Can’t lose them จำนวน 5.3%

10.Hibernating	จำนวน 13.7%

* **กลุ่มลูกค้าเป้าหมาย (กลุ่ม 8 9 และ 10)**
	
สำหรับกลุ่มที่คาดหวังลูกค้ายังคงใช้บริการต่อ ถ้าหากหายไปอาจจะส่งผลให้เสียยอดการใช้จ่ายของลูกค้า (รายได้บริษัท) ประมาณ 36% หากต้องเก็บลูกค้าไว้ต่อนั้นย่อมจะต้องทราบถึงข้อมูล Demographic ของลูกค้า ส่วนกลุ่ม 11.Lost เป็นกลุ่มที่ลูกค้ามีโอกาสจะปิดบัญชีนั้นซึ่งมีสัดส่วนไม่ได้มากนัก


* **การวิเคราะห์เกี่ยวกับข้อมูลอายุ**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/b52153fd-558c-435f-8947-120268dca25b)
![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/26dec728-15d6-44e8-8638-45975e05d949)
![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/c5ea5ce7-432a-4bd9-912f-3a43bc0bdb3b)
![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/5b1127ed-6a02-40fc-827e-647529953918)

08.At risk จำนวน

09.Can’t lose them 

10.Hibernating 

* **การวิเคราะห์เกี่ยวกับข้อมูลเพศ**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/6a7b8154-47f5-4725-b108-8db8a95512c0)

08.At risk จำนวน

09.Can’t lose them 

10.Hibernating 

* **การวิเคราะห์เกี่ยวกับข้อมูลระดับการศึกษา**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/e7b04d42-365c-4fb3-938d-1df99bde8870)

08.At risk จำนวน

09.Can’t lose them 

10.Hibernating 

* **การวิเคราะห์เกี่ยวกับข้อมูลสถานภาพสมรส**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/4d521b38-bb02-48df-93fc-8a620f7f82d5)

08.At risk จำนวน

09.Can’t lose them 

10.Hibernating 

* **การวิเคราะห์เกี่ยวกับข้อมูลรายได้**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/27799f90-3380-4d1a-aab7-b6b1988a4720)

08.At risk จำนวน

09.Can’t lose them 

10.Hibernating 

* **การวิเคราะห์เกี่ยวกับข้อมูลวงเงิน**

วางรูปตรงนี้

08.At risk จำนวน

09.Can’t lose them 

10.Hibernating

* **การวิเคราะห์เกี่ยวกับข้อมูลจำนวนทำธุรกรรม และยอดธุรกรรม**

วงรูปตรงนี้

08.At risk จำนวน

09.Can’t lose them 

10.Hibernating


**3.2 Descriptive Statistics**

สมมติฐานข้อมูล Demographic ได้แก่ อายุ จำนวนผลิตภัณฑ์ที่ลูกค้าถือครอง เพศ ระดับการศึกษา สถานภาพสมรส ช่วงรายได้ ของลูกค้า จะมีความสัมพันธ์ที่มีผลต่อการปิดบัญชีของลูกค้า

**_ภาพแสดงความสัมพันธ์ระหว่างเพศกับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/741b91f5-0967-4036-aa4d-c063edb930ff)

จากกราฟที่ X พบว่า เมื่อพิจารณาเพศกับการปิดบัญชีของลูกค้า ลูกค้าที่ทำการปิดบัญชีนั้นไม่ว่าจะเป็นเพศชายหรือหญิง จะมีการปิดบัญชีไม่ได้แตกต่างกันมากนัก หรือกล่าวได้ว่าเพศไม่ได้มีผลต่อการปิดบัญชี และอธิบายได้ว่าเพศของลูกค้ามีการใช้งานบัตรเครดิตและปิดบัญชีอย่างมีการแจงแจงแบบปกติ (Normal Distribution)

**_ภาพแสดงความสัมพันธ์ระหว่างจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองกับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/ffc9ede9-529e-451a-aba9-d7ca791625bf)

จากกราฟที่ X พบว่า ค่าเฉลี่ยของจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองไม่ได้มีผลต่อการปิดบัญชีของลูกค้า กล่าวคือข้อมูลลูกค้าที่ปิดบัญชีมีลักษณะเบ้ขวา อธิบายได้ว่าส่วนใหญ่แล้วลูกค้าที่ปิดบัญชีนั้นไม่ได้มีจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองมากเนื่องจากข้อมูลกระจุกอยู่ที่ 3 ผลิตภัณฑ์ต่อคนเท่านั้น หากเปรียบเทียบกับลูกค้าที่ยังใช้งานอยู่มีจำนวนผลิตภัณฑ์อยู่ที่ 4 ผลิตภัณฑ์ต่อคน

**_ภาพแสดงความสัมพันธ์ระหว่างจำนวนที่ติดต่อของลูกค้าในรอบ 12 เดือน กับการปิดบัญชีของลูกค้า_**

ว่างอยู่รอกราฟ

**_ภาพแสดงความสัมพันธ์ระหว่างสถานภาพสมรสกับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/179a1ac5-3e68-4179-9cde-06a66dc3a71c)

จากกราฟที่ X พบว่า สถานะภาพสมรสแต่ละกลุ่มมีกลุ่มลูกค้าที่ทำการปิดบัญชีตามสัดส่วนปริมาณลูกค้าในกลุ่มนั้นๆ ซึ่งสถานภาพสมรสอาจจะบ่งบอกเกี่ยวกับความสัมพันธ์ระหว่างสถานภาพสมรสกับการปิดบัญชีของลูกค้าได้ไม่ชัดเจนเท่ากับตัวแปรอื่นๆ เนื่องจากเป็นข้อมูลที่มีผลเกี่ยวข้องกับสภาพสังคมในปัจจุบัน

**_ภาพแสดงความสัมพันธ์ระหว่างระดับการศึกษา กับการปิดบัญชีของลูกค้า_**

ว่างอยู่รอกราฟ

**_ภาพแสดงความสัมพันธ์ระหว่างประเภทบัตร กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/6e21fc26-d04d-430d-b24b-9df32a403851)

**_ภาพแสดงความสัมพันธ์ระหว่างรายได้ กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/a3d9f2fc-72a0-4d51-82e3-6fed9c960022)

จากกราฟที่ X แสดงกลุ่มรายได้กับการปิดบัญชีของลูกค้า พบว่าลูกค้าทำการปิดบัญชี และใช้งานอยู่ จะมีช่วงรายได้ที่ค่อนข้างสัมพันธ์กัน นั้นหมายถึงไม่ว่าระดับรายได้กลุ่มไหนลูกค้าก็มีโอกาสที่จะทำการปิดบัญชี

**_ภาพแสดงความสัมพันธ์ระหว่างกลุ่มอายุ กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/425da319-ff80-45ec-84b6-b93477267f61)

จากกราฟที่ X พบว่าในกลุ่มอายุที่ใช้งานสูงอย่าง Generation Y และ X จะมีการปิดบัญชีที่สูงด้วย กล่าวคือมีความสัมพันธ์ไปในทางเดียวกัน ซึ่งแสดงให้เห็นว่ากลุ่ม Generation ไม่ได้มีผลต่อการปิดบัญชีของลูกค้า ละพบว่า Generation Z และ Baby Boomer ก็มีลักษณะข้อมูลเช่นเดียวกัน

**_กราฟแสดงกลุ่มอายุเปรียบเทียบกับยอดการใช้จ่าย_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/b38357dc-4796-479a-ac84-9fae99151695)

เมื่อวิเคราะห์กลุ่มอายุเปรียบเทียบกับยอดการใช้จ่าย พบว่า Generation X เป็นกลุ่มรายได้หลักของบริษัท แต่เมื่อมีการเปรียบเทียบการใช้จ่ายเฉลี่ยของ Generation พบว่า Generation Y และ X มีการใช้จ่ายไม่ได้แตกต่างกัน ซึ่งเป็นกลุ่มวัยทำงานมีรายได้เพียงพอก็จะมีการใช้จ่ายที่สูง


## **Part 4 : Summary**



## **Reference:**

[1] https://how-many-steps-inc.webflow.io/rfm-segmentation-overview

[2] https://www.putler.com/rfm-analysis/

[3] https://www.linkedin.com/pulse/rfm-analysis-important-revenue-growth-analytics-capability-kakas/
