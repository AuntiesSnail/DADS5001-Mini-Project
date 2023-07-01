# **Mini-Project: Credit Card Customer Churning**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/f58bdb8a-bb06-4593-ae38-888da814b33e)


## **Part 1 : Introduction**

Customer Churn Rate คืออัตราที่ลูกค้าหยุดใช้บริการในช่วงระยะเวลาใดเวลาหนึ่ง ซึ่งมีผลในการดำเนินธุรกิจที่ต้องการรักษาลูกค้าให้ใช้บริการกับเราให้ยาวนานและมากที่สุด ดังนั้น การป้องกันการยกเลิกใช้บริการของลูกค้าจึงมีความสำคัญมาก โดยการนำ Data ที่เป็นฐานลูกค้า มาจัดเป็น Customer Segment ด้วย RFM Model ร่วมกับข้อมูล Demographic เพื่อทำการวิเคราะห์และวางแผนการทำ Marketing กับลูกค้าในแต่ละกลุ่มให้ได้ตรงตามเป้าหมายและมีประสิทธิภาพมากที่สุด

ซึ่งชุดข้อมูลที่ใช้ในโปรเจคนี้ เป็นข้อมูลลูกค้า Consumer Credit Card Portfolio ของธนาคารแห่งหนึ่ง ที่ต้องการวิเคราะห์หาลูกค้าที่มีแนวโน้มจะยกเลิกการใช้บริการบัตรเครดิต เพื่อที่จะได้มอบการบริการที่ดีขึ้นและกระตุ้นการตัดสินใจของลูกค้าให้กลับมาใช้บริการอีกครั้ง


**1.1 ข้อมูล Dataset :**

>kaggle : [Predicting Credit Card Customer Segmentation](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m)

>Credit: The original authors – [Data Source](https://zenodo.org/record/4322342#.Y8OsBdJBwUE)

>License: [CC0 1.0 Universal (CC0 1.0) - Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)

>*No Copyright - You are allowed to copy, modify, distribute, and perform the work, even for commercial purposes, without the need for permission.*

**1.2 Hypothesis :**

การพิจารณา Customer Segment โดย RFM Model ร่วมกับข้อมูล Demographic จะช่วยทำให้เข้าใจพฤติกรรมของลูกค้าได้ดียิ่งขึ้น โดยเฉพาะกลุ่ม “At risk”, “Can’t lose them” และ “Hibernating” ที่มีแนวโน้มจะเลิกใช้บริการ ทำให้สามารถปรับกลยุทธ์ทางการตลาดและเสนอสิทธิพิเศษให้กับลูกค้าในแต่ละกลุ่มได้อย่างเหมาะสม


## **Part 2 : Exploratory Data Analysis (EDA)**

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

>1. จากข้อมูลทั้งหมด 23 ตัวแปร ซึ่งไม่มีตัวแปร Null และไม่ได้ใช้ตัวแปร Naive_Bayes1, Naive_Bayes2
 
>2. มีจำนวนทั้งหมด 3 ตัวแปร ได้แก่ Education_Level, Marital_Status, Income_Category พบข้อมูล Unknown ที่ไม่บงบอกสถานะที่แท้จริง

**2.2 Categorical and Numerical Variable:**

ในข้อมูลตัวแปร Attrition_Flag มี Attrited Customer (ลูกค้าที่ยกเลิกบริการ) 1,627 ราย (16.07%) และ Existing Customer (ลูกค้าที่ยังใช้บริการ) 8,500 ราย (83.93%)

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/079e7250-a130-47eb-a9dc-3da6ec44d755)

* **Categorical Data**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/61273a0d-f84a-43de-9dc2-f0e9658218b6)

>โดยบางตัวแปรข้อมูลระบุเป็น "Unknown"

>>Education_Level มีอยู่ 1,519 ราย เป็น Attrited Customer 256 ราย

>>Marital_Status มีอยู่ 749 ราย เป็น Attrited Customer 129 ราย

>>Income_Category มีอยู่ 1,112 ราย เป็น Attrited Customer 187 ราย

>Card_Category เป็นประเภท Blue เกือบทั้งหมดเลย มี 9,436 คน (93.18%)

* **Numerical Data**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/e471245b-e7cf-4db9-9da4-03acd6676aca)

>ข้อมูลค่อนข้าง clean แต่มีประเด็นที่น่าสงสัยคือ Avg_Utilization_Ratio เท่ากับ 0 ซึ่งลูกค้าไม่เคยมีการใช้บัตรเลย มีอยู่ 2,470 คน แต่ที่ตัวแปร Months_on_book กลับพบว่าเป็นลูกค้าใช้บริการมาแล้ว 13 เดือนขึ้นไปทุกคน และใช้บริการมากสุด 56 เดือน (เกือบ 5 ปี) แต่ไม่เคยใช้บัตร ซึ่งอาจหมายความได้ว่าลูกค้ามีการทำธุรกรรมกับธนาคารในรูปแบบอื่น

>ตัวแปร Attrition_Flag พบว่าลูกค้า ที่ Avg_Utilization_Ratio เท่ากับ 0 เป็น Attrited Customer 893 ราย ซึ่งมากกว่า 50.00 % ของ Attrited Customer ทั้งหมด

>ตัวแปรส่วนใหญ่มีลักษณะ "เบ้ขวา" เช่น ข้อมูลรายได้ (Income) พบว่าลูกค้าส่วนใหญ่มีรายได้ค่อนข้างน้อย ทำให้การกระจายของข้อมูลส่วนใหญ่กระจุกตัวอยู่ทางซ้าย


**2.3 การวิเคราะห์ RFM Model:**

RFM Analysis เป็น Model พื้นฐานเพื่อแบ่งกลุ่มลูกค้าหรือ Customer Segmentation ตามพฤติกรรมการใช้จ่าย โดยตัวแปรที่นำมาวิเคราะห์มี 3 ตัวแปรหลัก คือ R F และ M

  - Recency   (R) คือระยะเวลาที่ลูกค้าใช้งานบัตร ซึ่งจะใช้ตัวแปร Months_on_book 
  - Frequency (F) คือความถี่ในการใช้จ่ายของลูกค้า ซึ่งจะใช้ตัวแปร Total_Trans_C
  - Monetary  (M) คือยอดธุรกรรมทั้งหมด ซึ่งจะใช้ตัวแปร Total_Trans_Amt 


**_ตารางแสดงข้อมูลตัวอย่าง R F M_**


![table](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/d56f17a9-7c2d-4035-aea5-5fe3b514e60b)



**_การกำหนดเกณฑ์ให้คะแนน_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/4a84e2f8-0e61-4318-a938-525380c5b808)

จากรูปเป็นการแบ่งตามหลักการ Quintile โดยจะกำหนดเกณฑ์คะแนนตั้งแต่ 1 ถึง 5 และแบ่งจำนวนลูกค้าออกเป็นทั้งหมด 5 ช่วง เพื่อให้คะแนนในแต่ละด้านของ RFM Model วิธีการแบ่งกลุ่มคือการทำ Binning จากผลลัพธ์ที่ได้จาก .groupby() ข้างต้น

**_ตารางแสดงข้อมูลตัวอย่าง Rank R F M_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/20228d97-d64b-42f6-b7c1-d3a3331d9106)

ผลลัพธ์ที่ได้คือตัวเลข 1 ถึง 5 ที่เรียงลำดับตามพฤติกรรมที่เกิดขึ้น เช่น ถ้ามีค่า Recency น้อย ๆ จะถูกจัดอยู่ใน Binning ที่สูง หมายความว่าจำนวนวันที่เกิด Transaction ล่าสุด “ไม่นาน” โดยยิ่งลูกค้ามีการใช้บริการครั้งล่าสุดเร็วเท่าไหร่ ก็จะง่ายต่อการติดต่อกลับเพื่อ Engage ลูกค้าให้กลับมาใช้บริการอีกครั้ง

ถ้าหากมีค่า Frequency หรือมีความถี่ในการทำธุรกรรมบ่อยๆ จะถูกจัดอยู่ใน Binning ที่สูง (ไปในทิศทางเดียวกัน)  ส่วนค่าของ Monetary หรือยอดธุรกรรมมักจะสอดคล้องและไปเป็นในทิศทางเดียวกับค่า Frequency 

![Range](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/4c6b7035-7989-4fc4-ae19-94139ba77077)

หลังจากนั้นจัดทำ Segment เพื่อครอบคลุมเงื่อนไข RFM ทั้งหมด 125 (5x5x5) รูปแบบ 

และแบ่ง Customer Segment  ได้ทั้งหมด 11 Segment ได้แก่

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

**วิเคราะห์ข้อมูลในอดีตเบื้องต้นของ**

**_กราฟแสดง Customer Segment กับยอดธุรกรรมของ Attrited Customer_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/374b0f3b-6ae3-45e9-a230-86dffdf61cb5)


พบว่าข้อมูลลูกค้าที่ยกเลิกการใช้บัตรไปแล้ว กลุ่มลูกค้า 10.Hibernating, 04.Recent Customers และ 8.At Risk เป็นกลุ่มลูกค้าที่เคยมียอดธุรกรรมกับทางธนาคารสูงที่สุด 3 อันดับแรกตามลำดับ


## **Part 3 : Analyzing and Insight**

**3.1 Visualization**

* **RFM analysis**

**_กราฟแสดง Customer Segment (Tree Map)_** 

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/dc1c71e5-c343-4432-830a-a2ebb0cd1836)



**_กราฟเพื่อดูความสัมพันธ์ของ Recency Frequency Monetary ในกลุ่มลูกค้าแต่ละ Segment_**



![RFM](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/1ecaa137-c08d-41a1-8f8f-7f666d3106ef)




**_กราฟแสดงค่าเฉลี่ยของยอดธุรกรรม_**


![Heat](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/6e88823e-8550-4237-92db-3b3b0534bf07)



**_Focused Segment 8, 9, 10_** 


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

**_แสดงกลุ่มอายุ (Generation) กับ Segment_**	

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/055aada5-91cb-4452-9d8e-609fb0948966)

**_แสดงอายุกับ Segment 8, 9, 10_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/7172f64a-5650-4e57-8515-ba3626adf021)


08.At risk

ลูกค้าในกลุ่มนี้มีอายุเฉลี่ยที่ 53 ปี ซึ่งจัดเป็นกลุ่ม Generation X มากที่สุด

09.Can’t lose them 

ลูกค้าในกลุ่มนี้มีอายุเฉลี่ยที่ 55 ปี ซึ่งจัดเป็นกลุ่ม Generation X มากที่สุด 

10.Hibernating 

ลูกค้าในกลุ่มนี้มีอายุเฉลี่ยที่ 46 ปี ซึ่งจัดเป็นกลุ่ม Generation X และมีการกระจายของข้อมูลมากที่สุด


* **การวิเคราะห์เกี่ยวกับข้อมูลเพศ**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/1c82be57-06be-4aa7-9819-d4a7a1e95ce0)

08.At risk

ลูกค้าในกลุ่มนี้เป็นเพศหญิงมากกว่าเพศชาย
 
09.Can’t lose them

ลูกค้าในกลุ่มนี้เป็นเพศหญิงมากกว่าเพศชาย

10.Hibernating 

ลูกค้าในกลุ่มนี้เป็นเพศชายมากกว่าเพศหญิง


* **การวิเคราะห์เกี่ยวกับข้อมูลระดับการศึกษา**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/fdba163f-ee4f-4ed6-898d-ccc2595aaf83)

08.At risk

ลูกค้าในกลุ่มนี้เป็นผู้ที่จบการศึกษาระดับปริญาตรีมากที่สุดอย่างเห็นได้ชัด รองลงมาเป็นระดับ High School และ Uneducated และอื่นๆ ตามลำดับ

09.Can’t lose them

ลูกค้าในกลุ่มนี้เป็นผู้ที่จบการศึกษาระดับปริญาตรีมากที่สุด รองลงมาเป็นระดับ High School และ Uneducated และอื่นๆ ตามลำดับ 

10.Hibernating 


* **การวิเคราะห์เกี่ยวกับข้อมูลสถานภาพสมรส**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/25bd96eb-7779-46a5-bec9-5c895deeee86)

08.At risk

ลูกค้าในกลุ่มนี้มีสถานภาพ โสดเป็นส่วนใหญ่  และมีสถานภาพ สมรสแล้ว รองลงมา 

09.Can’t lose them

ลูกค้าในกลุ่มนี้มีสถานภาพ โสดและมีสถานภาพสมรสแล้ว ใกล้เคียงกัน และเป็นส่วนใหญ่

10.Hibernating 

ลูกค้าในกลุ่มนี้มีสถานภาพ สมรสแล้วเป็นส่วนใหญ่  และมีสถานโสด รองลงมา


* **การวิเคราะห์เกี่ยวกับข้อมูลรายได้**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/15d80947-04b8-4eea-81f3-d1f663071825)

08.At risk

ลูกค้าในกลุ่มนี้มีระดับรายได้ที่ค่อนข้างใกล้เคียงกัน แต่พบมีระดับรายได้ น้อยกว่า $40K มากที่สุดอย่างเห็นได้ชัด  

09.Can’t lose them

ลูกค้าในกลุ่มนี้มีระดับรายได้ที่ค่อนข้างใกล้เคียงกัน 

10.Hibernating 

ลูกค้าในกลุ่มนี้มีระดับรายได้ที่ค่อนข้างใกล้เคียงกัน แต่พบมีระดับรายได้ น้อยกว่า $40K มากที่สุด  


* **การวิเคราะห์เกี่ยวกับข้อมูลวงเงิน**
  
**_แสดงข้อมูลวงเงินบัตรเครดิต กับ Segment 8, 9, 10_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/ef70b185-b7c6-4336-b8fe-852bde319499)

08.At risk, 09.Can’t lose them และ 10.Hibernating 

ลูกค้าทั้ง 3 กลุ่มนี้ส่วนใหญ่มีจำนวนวงเงินที่ได้รับการอนุมัติจากธนาคารเป็นในลักษณะเบ้ขวา กล่าวคือได้รับการอนุมัติวงเงินกระจุกเนื่องจากฐานรายได้ที่ใกล้เคียงกัน


* **การวิเคราะห์เกี่ยวกับข้อมูลจำนวนทำธุรกรรม และยอดธุรกรรม**

**_แสดงข้อมูลจำนวนทำธุรกรรม กับ Segment 8, 9, 10_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/a19ee6db-016a-46f9-a272-c8f3e42f3151)


**_แสดงข้อมูลยอดธุรกรรมกับ Segment 8, 9, 10_**


![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/915f3622-d50e-4243-afdd-c4b823dd86a1)


08.At risk

ลูกค้ากลุ่มนี้มีความถี่ในการทำธุรกรรมและยอดธุรกรรมสูง กลาง และต่ำ

09.Can’t lose them 

ลูกค้ากลุ่มนี้มีความถี่ในการทำธุรกรรมสูง ในขณะที่ยอดธุรกรรมอยู่ในระดับกลาง และต่ำ

10.Hibernating 

ลูกค้ากลุ่มนี้มีความถี่ในการทำธุรกรรมและยอดธุรกรรมต่ำ


**3.2 Descriptive Statistics**

สมมติฐานข้อมูล Demographic ได้แก่ อายุ จำนวนผลิตภัณฑ์ที่ลูกค้าถือครอง เพศ ระดับการศึกษา สถานภาพสมรส ช่วงรายได้ ของลูกค้า จะมีความสัมพันธ์ที่มีผลต่อการปิดบัญชีของลูกค้า

**_ภาพแสดงความสัมพันธ์ระหว่างเพศกับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/db89d250-f13a-4909-bd36-6d66e7270789)

พบว่าลูกค้าที่ทำการปิดบัญชีส่วนใหญ่แล้วเป็นเพศหญิง ซึ่งเพศไม่ได้มีผลต่อการปิดบัญชีเนื่องจากมีสัดส่วนที่ใกล้เคียงกัน

**_ภาพแสดงความสัมพันธ์ระหว่างอายุกับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/c8ea0174-384e-4c40-a475-1d4ff4a37891)

พบว่า เมื่อพิจารณาอายุกับการปิดบัญชีของลูกค้า ลูกค้าที่ทำการปิดบัญชีนั้นไม่ว่าจะอายุเท่าไหร่ จะมีการปิดบัญชีไม่ได้แตกต่างกันมากนัก และอธิบายได้ว่าอายุของลูกค้ามีการใช้งานบัตรเครดิตและปิดบัญชีอย่างมีการแจงแจงแบบปกติ (Normal Distribution)

**_ภาพแสดงความสัมพันธ์ระหว่างกลุ่มอายุ (Generation) กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/2cdf89f9-33f5-456b-afa9-195f3753d685)

พบว่าในกลุ่มอายุที่ใช้งานสูงอย่าง Generation X , Y และจะมีการปิดบัญชีที่สูงด้วย ส่วน Baby Boomer มีการใช้งานน้อย แต่จะมีการปิดบัญชีที่สูงเท่าๆ กับ Generation  X , Y  สำหรับ Generation Z เป็นกลุ่มที่มีการใช้งานน้อยและมีการปิดบัญชีน้อยด้วยเช่นกัน

**_กราฟแสดงกลุ่มอายุเปรียบเทียบกับยอดการใช้จ่าย_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/b38357dc-4796-479a-ac84-9fae99151695)

เมื่อวิเคราะห์กลุ่มอายุเปรียบเทียบกับยอดการใช้จ่าย พบว่า Generation X เป็นกลุ่มรายได้หลักของบริษัท แต่เมื่อมีการเปรียบเทียบการใช้จ่ายเฉลี่ยของ Generation พบว่า Generation Y และ X มีการใช้จ่ายไม่ได้แตกต่างกัน ซึ่งเป็นกลุ่มวัยทำงานมีรายได้เพียงพอก็จะมีการใช้จ่ายที่สูง

**_ภาพแสดงความสัมพันธ์ระหว่างจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองกับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/ffc9ede9-529e-451a-aba9-d7ca791625bf)

พบว่า ค่าเฉลี่ยของจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองไม่ได้มีผลต่อการปิดบัญชีของลูกค้า กล่าวคือข้อมูลลูกค้าที่ปิดบัญชีมีลักษณะเบ้ขวา อธิบายได้ว่าส่วนใหญ่แล้วลูกค้าที่ปิดบัญชีนั้นไม่ได้มีจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองมากเนื่องจากข้อมูลกระจุกอยู่ที่ 3 ผลิตภัณฑ์ต่อคนเท่านั้น หากเปรียบเทียบกับลูกค้าที่ยังใช้งานอยู่มีจำนวนผลิตภัณฑ์อยู่ที่ 4 ผลิตภัณฑ์ต่อคน

**_ภาพแสดงความสัมพันธ์ระหว่างสถานภาพสมรสกับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/179a1ac5-3e68-4179-9cde-06a66dc3a71c)

พบว่า สถานะภาพสมรสแต่ละกลุ่มมีกลุ่มลูกค้าที่ทำการปิดบัญชีตามสัดส่วนปริมาณลูกค้าในกลุ่มนั้นๆ ซึ่งสถานภาพสมรสอาจจะบ่งบอกเกี่ยวกับความสัมพันธ์ระหว่างสถานภาพสมรสกับการปิดบัญชีของลูกค้าได้ไม่ชัดเจนเท่ากับตัวแปรอื่นๆ เนื่องจากเป็นข้อมูลที่มีผลเกี่ยวข้องกับสภาพสังคมในปัจจุบัน

**_ภาพแสดงความสัมพันธ์ระหว่างระดับการศึกษาและรายได้ กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/7338798c-23ac-4cb4-b4fc-ed74824398b6)

พบว่าลูกค้าส่วนใหญ่เป็นกลุ่มลูกค้าที่มีระดับรายได้น้อยกว่า $40K และมีระดับการศึกษาเป็นระดับปริญญา

**_ภาพแสดงความสัมพันธ์ระหว่างประเภทบัตร กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/4831b7c0-d8c9-4b7d-ae44-c52de983dd85)

พบว่าลูกส่วนใหญ่จะเป็นผู้ถือบัตรประเภท Blue ในขณะที่ลูกค้าที่ทำการปิดบัตรจะเป็นบัตรประเภท Platinum

**_ภาพแสดงความสัมพันธ์ระหว่างรายได้ กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/f5f55259-3295-44af-baab-d313cf0b54af)

แสดงกลุ่มรายได้กับการปิดบัญชีของลูกค้า พบว่าลูกค้าทำการปิดบัญชี และใช้งานอยู่ จะมีช่วงรายได้ที่ค่อนข้างใกล้เคียงกัน 

**_ภาพแสดงความสัมพันธ์ระหว่างวงเงิน กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/854e385a-dfd3-476e-a4e0-4091149f1a70)

แสดงความสัมพันธ์ระหว่างวงเงิน กับการปิดบัญชีของลูกค้า ซึ่งจะเห็นได้ว่าส่วนใหญ่ที่ทำการปิดบัญชีจะมีระดับวงเงินที่น้อย

**_ภาพแสดงความสัมพันธ์ระหว่างจำนวนผู้ที่อยู่ในความอุปการะ กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/1d9d9e44-96a7-4b1a-aa66-b9b0c5f9df01)

แสดงความสัมพันธ์ระหว่างจำนวนผู้ที่อยู่ในความอุปการะ กับการปิดบัญชีของลูกค้า ลูกค้าที่มีการปิดบัญชีส่วนใหญ่จะมีจำนวนผู้ที่อยู่ในความอุปการะอยู่ที่ 3 คน แต่ภาพรวมจะมีการปิดบัญชีเป็นไปอย่างปกติ

**_ภาพแสดงความสัมพันธ์ระหว่างจำนวนเดือนที่ลูกค้าใช้บริการ กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/1a6dada5-575f-47a2-9c58-ac83267a7b54)

แสดงความสัมพันธ์ระหว่างจำนวนเดือนที่ลูกค้าใช้บริการ กับการปิดบัญชีของลูกค้า พบว่าลูกค้าส่วนใหญ่ที่ปิดบัญชีจะเป็นกลุ่มลูกค้าที่ถือบัตรกับธนาคารมานานระดับหนึ่ง เป็นลูกค้าใหม่ค่อนข้างน้อย

**_ภาพแสดงความสัมพันธ์ระหว่างยอดธุรกรรม กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/dda78725-1dbc-41bb-acc4-282605f91fbf)

ลูกค้าที่ปิดบัญชีไปแล้วมีการกระจุกตัวไม่ได้มากนัก เมื่อเทียบกับลูกค้าที่ยังใช้บริการบัตรอยู่ถึงแม้ยอดธุรกรรมจะไม่มากนักแต่ก็ยังมีความถี่ในการทำธุรกรรมค่อนข้างสูง และจากกราฟพบว่าลูกค้าที่มีจำนวนธุรกรรมและยอดการใช้งานสูงจะไม่มีแนวโน้มการปิดบัตรเลย

**_ภาพแสดงความสัมพันธ์ระหว่างจำนวนเดือนที่ลูกค้าไม่มีการเคลื่อนไหว และการติดต่อ ใน 12 เดือนล่าสุด กับการปิดบัญชีของลูกค้าภาพแสดงความสัมพันธ์ระหว่างจำนวนเดือนที่ลูกค้าไม่มีการเคลื่อนไหว และการติดต่อ ใน 12 เดือนล่าสุด กับการปิดบัญชีของลูกค้า_**

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/28b012d4-0b59-478a-8a0d-1f0a55f6122f)

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/06e87e13-b842-45c2-82fa-e8b1ae5201fe)

ลูกค้าที่มีการทำธุรกรรมมานานและไม่มีการติดต่อมานานมีแนวโน้มที่จะเลิกใช้มากที่สุด


## **Part 4 : Summary**

* **สรุปกลุ่มลูกค้าเป้าหมาย (กลุ่ม 8 9 และ 10)**


|ตัวแปรที่มีนัยสำคัญ|08.At risk|09.Can’t lose them|10.Hibernating|
|--------------|----------|-------------------|---------------|
|อายุเฉลี่ย (ปี)|53	|55|46
|เพศ|หญิง|หญิง|หญิง|
|ฐานรายได้ส่วนใหญ่|น้อยกว่า $40K|น้อยกว่า $40K	|น้อยกว่า $40K|
|การเคลื่อนไหวทางธุรกรรม	|2.40	|2.29	|2.47|
|ยอดธุรกรรมเฉลี่ย (USD)	| 5,793.98 |	 7,741.00| 	 2,572.25 |
|จำนวนลูกค้า	|15.10%	|5.30%	|13.70%|
|ระดับความสำคัญ|	1|	2|	3|

		
08.At risk จากการวิเคราะห์พบว่าลูกค้าในกลุ่มนี้มีอายุเฉลี่ยที่ 53 ปี ส่วนใหญ่เป็นเพศหญิง ซึ่งคาดว่าเป็นวัยที่มีการใช้จ่ายสูง มีสถานโสดมากที่สุด ดังนั้น ทางธนาคารควรต้องรีบติดต่อลูกค้าในกลุ่มนี้ เพื่อที่จะกระตุ้นยอดบัตรเครดิต

09.Can’t lose them จากการวิเคราะห์พบว่าลูกค้าในกลุ่มนี้มีอายุเฉลี่ยที่ 55 ปี ส่วนใหญ่เป็นเพศหญิง และเป็นลูกค้าที่เคยสร้างรายได้ให้กับธนาคารมานาน ทางธนาคารจึงควรให้ความสำคัญในการดึงลูกค้ากลุ่มนี้กลับมาเป็นพิเศษ ไม่ควรเสียลูกค้ากลุ่มนี้ให้กับคู่แข่ง

10.Hibernating จากการวิเคราะห์พบว่าลูกค้าในกลุ่มนี้มีอายุเฉลี่ยที่ 46 ปี ส่วนใหญ่เป็นเพศหญิง ลูกค้ากลุ่มนี้มีความน่าสนใจหากต้องการรักษาฐานลูกค้า


* **สรุปแนวโน้มการเลิกใช้บริการด้วยความสัมพันธ์ของตัวแปร**

**Categorical Variable:**

Generation: 	Generation X , Y และ Baby Boomer จะมีการปิดบัญชีที่สูง ส่วน Generation Z เป็นกลุ่มที่มีการปิดบัญชีน้อย

เพศ: 		เพศหญิงเป็นเพศที่มีการใช้บริการบัตรเครดิตสูงสุดซึ่งมีความสัมพันธ์กับเหตุผลที่จะเลิกใช้

รายได้: 		ช่วงรายได้ไม่ได้มีผลกับการเลิกใช้บัตรเครดิต 

อื่นๆ: 		ระดับการศึกษา: สถานภาพสมรส ไม่ได้เป็นสาเหตุในการเลิกใช้บัตรเครดิต

 
**Numerical Variable:**

วงเงิน:		วงเงินเงินบัตรเครดิตมีต่อการเลิกใช้บริการบัตรเครดิตของลูกค้า เนื่องจากไม่เพียงพอต่อความต้องการใช้งาน ยิ่งวงเงินน้อยลูกค้ายิ่งมีแนวโน้มยกเลิกการใช้งานน้อย

การเคลื่อนไหว: 	ลูกค้าที่มีการทำธุรกรรมมานานและไม่มีการติดต่อมานานมีแนวโน้มที่จะเลิกใช้มากที่สุด

อื่นๆ: 		จำนวนผู้อยู่ในความอุปการะ จำนวนธุรกรรม ยอดธุรกรรม วงเงินคงเหลือ เป็นต้น มีผลต่อแนวโน้มต่อการยกเลิกบัตรเครดิตต่ำ


* **Action plan**


![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/136181947/1ef36780-19e8-49dd-b04c-bfa32c3915e5)


จากการทำ Customer Segment เราควรออกแบบโปรโมชั่นหรือแคมเปญการตลาดที่เหมาะสมกับกลุ่มลูกค้า 

**กลุ่ม 08.At risk**

ควรทำการตลาดด้วยข้อเสนอแบบจำกัดเวลา เพื่อกระตุ้นยอดใช้จ่าย โดยนำเสนอโปรโมชั่น เช่น
โปรโมชั่น: " Welcome Your Rewards!”
- รับคะแนนสะสมสองเท่าสำหรับยอดการใช้จ่ายทุกครั้งในช่วงระยะเวลาส่งเสริมการขาย
- หากสามารถสะสมยอดการใช้จ่ายได้ตามเป้าหมายที่กำหนดจะได้รับรางวัลโบนัสพิเศษ
 	
**กลุ่ม 09.Can’t lose them**

ควรวางกลยุทธ์ให้ลูกค้าในกลุ่มนี้มีความสนใจที่ต้องการใช้งาน และรู้สึกว่าเป็นลูกค้าคนสำคัญ เนื่องจากมีจำนวนการใช้งานน้อยแต่ยอดธุรกรรมสูง ควรทำโปรโมชั่นที่ดึงดูดความสนใจ จนต้องกลับมาใช้อีกครั้ง เช่น
โปรโมชั่น: "Exclusive Rewards Membership"
- เพิ่มอัตราการรับรางวัลเพื่อการสะสมคะแนนที่เร็วขึ้น
- มอบประสบการณ์วีไอพี ในการกิจกรรมพิเศษ ส่วนลดร้านอาหาร และสถานที่ท่องเที่ยว
- มอบข้อเสนอเฉพาะบุคคล ส่วนลดพิเศษ และการสนับสนุนลูกค้าตามลำดับความสำคัญ

**กลุ่ม 10.Hibernating**

เนื่องจากลูกค้าในกลุ่มนี้ถึงแม้จะมียอดการทำธุรกรรมค่อนข้างต่ำ แต่มีความถี่ในการใช้จ่ายสูง ลูกค้ากลุ่มนี้มีความน่าสนใจหากต้องการรักษาฐานลูกค้า ควรนำเสนอโปรโมชั่น เช่น
	โปรโมชั่น: " Come Back and Save! "
- ยื่นข้อเสนอในการยกเว้นค่าธรรมเนียมรายปีสำหรับปีถัดไป
- รับคะแนนโบนัสสำหรับทุกธุรกรรมในช่วงโปรโมชั่น
- มอบส่วนลดและข้อเสนอสุดพิเศษจากร้านค้าพันธมิตร
- อัพเดทข้อมูลและกิจกรรมผ่านอีเมลหรือ SMS แจ้งเตือนรายเดือน 


## **Reference:**
[1] https://how-many-steps-inc.webflow.io/rfm-segmentation-overview

[2] https://www.putler.com/rfm-analysis/

[3] https://www.linkedin.com/pulse/rfm-analysis-important-revenue-growth-analytics-capability-kakas/
