# **Mini-Project: Credit Card Customer Churning**

## **Part 1 : Introduction**

Customer Churn Rate คืออัตราที่ลูกค้าหยุดใช้บริการในช่วงระยะเวลาใดเวลาหนึ่ง ซึ่งมีผลในการดำเนินธุรกิจที่ต้องการรักษาลูกค้าให้ใช้บริการกับเราให้ยาวนานและมากที่สุด ดังนั้น การป้องกันการยกเลิกใช้บริการของลูกค้าจึงมีความสำคัญมาก โดยการนำ Data ที่เป็นฐานลูกค้า มาจัดเป็น Customer Segment ด้วย RFM Model ร่วมกับข้อมูล Demographic เพื่อทำการวิเคราะห์และวางแผนการทำ Marketing กับลูกค้าในแต่ละกลุ่มให้ได้ตรงตามเป้าหมายและมีประสิทธิภาพมากที่สุด

ซึ่งชุดข้อมูลที่ใช้ในโปรเจคนี้ เป็นข้อมูลลูกค้า Consumer Credit Card Portfolio ของธนาคารแห่งหนึ่ง ที่ต้องการวิเคราะห์หาลูกค้าที่มีแนวโน้มจะยกเลิกการใช้บริการบัตรเครดิต เพื่อที่จะได้มอบการบริการที่ดีขึ้นและกระตุ้นการตัดสินใจของลูกค้าให้กลับมาใช้บริการอีกครั้ง

**ข้อมูล Dataset :**

* kaggle : Predicting Credit Card Customer Segmentation

* Credit: The original authors – Data Source

* License: CC0 1.0 Universal (CC0 1.0) - Public Domain Dedication

* No Copyright - You are allowed to copy, modify, distribute, and perform the work, even for commercial purposes, without the need for permission

**Hypothesis :**

การพิจารณา Customer Segment โดย RFM Model ร่วมกับข้อมูล Demographic จะช่วยทำให้เข้าใจพฤติกรรมของลูกค้าได้ดียิ่งขึ้น โดยเฉพาะกลุ่ม “At risk”, “Can’t lose them” และ “Hibernating” ที่มีแนวโน้มจะเลิกใช้บริการ ทำให้สามารถปรับกลยุทธ์ทางการตลาดและเสนอสิทธิพิเศษให้กับลูกค้าในแต่ละกลุ่มได้อย่างเหมาะสม


## **Part 2 : Exploration Data Analysis (EDA)**

**2.1 Review Data:**

ข้อมูลลูกค้าทั้งหมด 10,127 ราย ประกอบด้วย 21 ตัวแปร ได้แก่

* CLIENTNUM		รหัสลูกค้าธนาคาร	
* Attrition_Flag	แสดงสถานะการใช้บริการของลูกค้า	
* Customer_Age		อายุลูกค้า	Data Demographic
* Gender		เพศลูกค้า	Data Demographic
* Dependent_count	จำนวนผู้ที่อยู่ในความอุปการะ	
* Education_Level	ระดับการศึกษาDemographic
* Marital_Status	สถานภาพสมรส	Demographic
* Income_Category	ช่วงรายได้ของลูกค้า	Demographic
* Card_Category	ประเภทบัตรของลูกค้า	
* Months_on_book	จำนวนเดือนที่ลูกค้าใช้บริการ	
* Total_Relationship_Count	จำนวนผลิตภัณฑ์ที่ลูกค้าถือครองกับธนาคาร	Demographic
* Months_Inactive_12_mon	จำนวนเดือนที่ลูกค้าไม่มีการเคลื่อนไหว ใน 12 เดือนล่าสุด	
* Contacts_Count_12_mon	จำนวนครั้งที่มีการติดต่อ ใน 12 เดือนล่าสุด	
* Credit_Limit	วงเงินบัตรเครดิต	
* Total_Revolving_Bal	ยอดค้างชำระ	
* Avg_Open_To_Buy	วงเงินเครดิตที่พร้อมใช้สำหรับการซื้อสินค้าหรือทำธุรกรรมใหม่ (คำนวณค่าเฉลี่ยของวงเงินเครดิตที่เปิดให้ใช้งานในระยะเวลา 12 เดือน)	
* Total_Amt_Chng_Q4_Q1	ยอดใช้จ่าย Q4 เทียบกับ Q1	
* Total_Trans_Amt	ยอดใช้จ่ายทั้งหมด ใน 12 เดือนล่าสุด	
* Total_Trans_Ct	จำนวนรายการทั้งหมด ใน 12 เดือนล่าสุด	
* Total_Ct_Chng_Q4_Q1	จำนวนรายการ Q4 เทียบกับ Q1	
* Avg_Utilization_Ratio	ยอดการใช้จ่ายต่อวงเงิน เป็นอัตราเท่าไหร่ กำหนดให้ 1 คือใช้จ่ายเต็มวงเงิน และ 0 คือไม่มีการใช้จ่าย คิดเป็นค่าเฉลี่ยจากทั้งหมด 	

**2.2 Categorical and Numerical Variable:**

ในข้อมูลตัวแปร Attrition_Flag มี Attrited Customer (ลูกค้าที่ยกเลิกบริการ) 1,627 ราย (16.07%) และ Existing Customer (ลูกค้าที่ยังใช้บริการ) 8,500 ราย (83.93%)

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/2a8dcbc7-a2a1-4e80-9712-0bb6af83fe73)


**2.3 การวิเคราะห์ RFM Model:**

RFM Analysis เป็น Model พื้นฐานเพื่อแบ่งกลุ่มลูกค้าหรือ Customer Segmentation ตามพฤติกรรมการใช้จ่าย โดยตัวแปรที่นำมาวิเคราะห์มี 3 ตัวแปรหลัก คือ R F และ M

* Recency   (R) คือระยะเวลาที่ลูกค้าใช้งานบัตร ซึ่งจะใช้ตัวแปร Months_on_book 
* Frequency (F) คือความถี่ในการใช้จ่ายของลูกค้า ซึ่งจะใช้ตัวแปร Total_Trans_C
* Monetary  (M) คือยอดธุรกรรมทั้งหมด ซึ่งจะใช้ตัวแปร Total_Trans_Amt 

*ตารางแสดงข้อมูลตัวอย่าง R F M*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/c0f8c169-57fb-48d7-8918-bbe1bde050bd)

*การกำหนดเกณฑ์ให้คะแนน*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/4a84e2f8-0e61-4318-a938-525380c5b808)

จากรูปเป็นการแบ่งตามหลักการ Quintile โดยจะกำหนดเกณฑ์คะแนนตั้งแต่ 1 ถึง 5 และแบ่งจำนวนลูกค้าออกเป็นทั้งหมด 5 ช่วง เพื่อให้คะแนนในแต่ละด้านของ RFM Model วิธีการแบ่งกลุ่มคือการทำ Binning จากผลลัพธ์ที่ได้จาก .groupby() ข้างต้น

*ตารางแสดงข้อมูลตัวอย่าง Rank R F M*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/20228d97-d64b-42f6-b7c1-d3a3331d9106)

ผลลัพธ์ที่ได้คือตัวเลข 1 ถึง 5 ที่เรียงลำดับตามพฤติกรรมที่เกิดขึ้น เช่น ถ้ามีค่า Recency น้อย ๆ จะถูกจัดอยู่ใน Binning ที่สูง หมายความว่าจำนวนวันที่เกิด Transaction ล่าสุด “ไม่นาน” โดยยิ่งลูกค้ามีการใช้บริการครั้งล่าสุดเร็วเท่าไหร่ ก็จะง่ายต่อการติดต่อกลับเพื่อ Engage

ถ้าหากมีค่า Frequency หรือมีความถี่ในการทำธุรกรรมบ่อยๆ จะถูกจัดอยู่ใน Binning ที่สูง (ไปในทิศทางเดียวกัน)  ส่วนค่าของ Monetary หรือยอดธุรกรรมมักจะสอดคล้องและไปเป็นในทิศทางเดียวกับค่า Frequency 

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/afe44d47-921a-417f-aa44-78f1e5cbdf2e)

หลังจากนั้นจัดทำ Segment เพื่อครอบคลุมเงื่อนไข RFM ทั้งหมด 125 (5x5x5) รูปแบบ และแบ่ง Customer Segment  ได้ทั้งหมด 11 Segment ได้แก่

* 01.Champions		Bought recently, buy often and spend the most
* 02.Loyal Customers	Spend good money. Responsive to promotions
* 03.Potential Loyalist	Recent customers, spent good amount, bought more than once
* 04.Recent Customers	Bought more recently, but not often
* 05.Promising		Recent shoppers, but haven’t spent much
* 06.Need Attention	Above average recency, frequency & monetary values
* 07.About to Sleep	Below average recency, frequency & monetary values
* 08.At Risk		Spent big money, purchased often but long time ago
* 09.Can’t Lose Them	Made big purchases and often, but long time ago
* 10.Hibernating	Low spenders, low frequency, purchased long time ago
* 11.Lost		Lowest recency, frequency & monetary scores

## **Part 3 : Insight**

**3.1 Descriptive Statistics**

สมมติฐานข้อมูล Demographic ได้แก่ อายุ จำนวนผลิตภัณฑ์ที่ลูกค้าถือครอง เพศ ระดับการศึกษา สถานภาพสมรส ช่วงรายได้ ของลูกค้า จะมีความสัมพันธ์ที่มีผลต่อการปิดบัญชีของลูกค้า

*ภาพแสดงความสัมพันธ์ระหว่างเพศกับการปิดบัญชีของลูกค้า*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/8f58ff9c-86ef-4dc9-80f7-a620d117e6ca)
*กราฟที่ X แสดงความสัมพันธ์ระหว่างเพศกับการปิดบัญชีของลูกค้า*

จากกราฟที่ X พบว่า เมื่อพิจารณาเพศกับการปิดบัญชีของลูกค้า ลูกค้าที่ทำการปิดบัญชีนั้นไม่ว่าจะเป็นเพศชายหรือหญิง จะมีการปิดบัญชีไม่ได้แตกต่างกันมากนัก หรือกล่าวได้ว่าเพศไม่ได้มีผลต่อการปิดบัญชี และอธิบายได้ว่าเพศของลูกค้ามีการใช้งานบัตรเครดิตและปิดบัญชีอย่างมีการแจงแจงแบบปกติ (Normal Distribution)

*ภาพแสดงความสัมพันธ์ระหว่างจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองกับการปิดบัญชีของลูกค้า*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/6f8e8825-f340-4bb3-9ff0-8a2b8534d7e5)
*กราฟที่ X แสดงความสัมพันธ์ระหว่างจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองกับการปิดบัญชีของลูกค้า*

จากกราฟที่ X พบว่า ค่าเฉลี่ยของจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองไม่ได้มีผลต่อการปิดบัญชีของลูกค้า กล่าวคือข้อมูลลูกค้าที่ปิดบัญชีมีลักษณะเบ้ขวา อธิบายได้ว่าส่วนใหญ่แล้วลูกค้าที่ปิดบัญชีนั้นไม่ได้มีจำนวนผลิตภัณฑ์ที่ลูกค้าถือครองมากเนื่องจากข้อมูลกระจุกอยู่ที่ 3 ผลิตภัณฑ์ต่อคนเท่านั้น หากเปรียบเทียบกับลูกค้าที่ยังใช้งานอยู่มีจำนวนผลิตภัณฑ์อยู่ที่ 4 ผลิตภัณฑ์ต่อคน

*ภาพแสดงความสัมพันธ์ระหว่างสถานภาพสมรสกับการปิดบัญชีของลูกค้า*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/72668ab1-a11e-4ca2-b815-6546901b0c37)
*กราฟที่ X แสดงความสัมพันธ์ระหว่างสถานภาพสมรสกับการปิดบัญชีของลูกค้า*

จากกราฟที่ X พบว่า สถานะภาพสมรสแต่ละกลุ่มมีกลุ่มลูกค้าที่ทำการปิดบัญชีตามสัดส่วนปริมาณลูกค้าในกลุ่มนั้นๆ ซึ่งสถานภาพสมรสอาจจะบ่งบอกเกี่ยวกับความสัมพันธ์ระหว่างสถานภาพสมรสกับการปิดบัญชีของลูกค้าได้ไม่ชัดเจนเท่ากับตัวแปรอื่นๆ เนื่องจากเป็นข้อมูลที่มีผลเกี่ยวข้องกับสภาพสังคมในปัจจุบัน

*ภาพแสดงความสัมพันธ์ระหว่างรายได้ กับการปิดบัญชีของลูกค้า*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/e4af3fb4-b17e-4085-91a7-38c46b02123e)
*กราฟที่ X แสดงความสัมพันธ์ระหว่างรายได้ กับการปิดบัญชีของลูกค้า*

จากกราฟที่ X แสดงกลุ่มรายได้กับการปิดบัญชีของลูกค้า พบว่าลูกค้าทำการปิดบัญชี และใช้งานอยู่ จะมีช่วงรายได้ที่ค่อนข้างสัมพันธ์กัน นั้นหมายถึงไม่ว่าระดับรายได้กลุ่มไหนลูกค้าก็มีโอกาสที่จะทำการปิดบัญชี

*ภาพแสดงความสัมพันธ์ระหว่างกลุ่มอายุ กับการปิดบัญชีของลูกค้า*

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/831b1b5d-38b9-4041-9c2c-b7c436d789ef)
*กราฟที่ Xแสดงความสัมพันธ์ระหว่างกลุ่มอายุ กับการปิดบัญชีของลูกค้า*

จากกราฟที่ X พบว่าในกลุ่มอายุที่ใช้งานสูงอย่าง Generation Y และ X จะมีการปิดบัญชีที่สูงด้วย กล่าวคือมีความสัมพันธ์ไปในทางเดียวกัน ซึ่งแสดงให้เห็นว่ากลุ่ม Generation ไม่ได้มีผลต่อการปิดบัญชีของลูกค้า ละพบว่า Generation Z และ Baby Boomer ก็มีลักษณะข้อมูลเช่นเดียวกัน

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/d9d1a7ba-10e9-41d6-a626-0fa79fe2752f)
*กราฟที่ X กลุ่มอายุเปรียบเทียบกับยอดการใช้จ่าย*

เมื่อวิเคราะห์กลุ่มอายุเปรียบเทียบกับยอดการใช้จ่าย พบว่าการใช้จ่ายของแต่ละกลุ่มอายุโดยเฉลี่ยในแต่ละกลุ่มเป็นไปในทิศทางที่ค่อนข้างปกติของข้อมูล กล่าวคือกลุ่ม Generation Y และ X ซึ่งเป็นกลุ่มวัยทำงานมีรายได้เพียงพอก็จะมีการใช้จ่ายที่สูง

**3.2 Visualization**

* RFM Analysis
  
Attrition_Flag คือ การปิดบัญชีของลูกค้า ซึ่งเราจะใช้ตัวแปรนี้เป็น label ในการจำแนกประเภทลูกค้า ในการทำ model RFM เพื่อดูกลุ่มที่ใช้งานอยู่ว่ามีคนไหนที่จะไม่ใช้งานต่อ 

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/26be9de3-8a6c-453f-81c0-de26bcfd42f9)

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/6cd84fe4-c57b-4d68-8a8b-855428bc3840)

จากการจัด Segment ด้วยวิธี RFM พบว่า กลุ่มลูกค้า ที่จัดอยู่ใน Segment 8,9,10 เป็นกลุ่มลูกค้าที่มีโอกาสที่จะเลิกใช้บริการบัตร  ส่วน Segment 11 ซึ่งเป็นกลุ่มที่จะเลิกใช้บริการ ซึ่งแบ่งเป็นสัดส่วนดังภาพด้านล่าง

![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/e975078f-b2d2-46e1-a455-c1cda7be00dd)

* Churning
  
จากการทำ Model RFM พบว่า กลุ่มที่ใช้งานอยู่มีโอกาสเป็นกลุ่ม lost ถึง 6.7% และ กลุ่มที่เราสามารถนำเสนอเพื่อกระตุ้น และคาดหวังลูกค้ายังคงใช้บริการต่อไป ได้แก่ 

**- 08.At risk		จำนวน 15.1%**

**- 09.Can’t lose them	จำนวน 5.3%**

**- 10.Hibernating	จำนวน 13.7%**

สำหรับกลุ่มที่คาดหวังลูกค้ายังคงใช้บริการต่อ ถ้าหากหายไปอาจจะส่งผลให้เสียยอดการใช้จ่ายของลูกค้า (รายได้บริษัท) ประมาณ 36% หากต้องเก็บลูกค้าไว้ต่อนั้นย่อมจะต้องทราบถึงข้อมูล Demographic ของลูกค้า ส่วนกลุ่ม 11.Lost เป็นกลุ่มที่ลูกค้ามีโอกาสจะปิดบัญชีนั้นซึ่งมีสัดส่วนไม่ได้มากนัก

* TreeMap

สำหรับกลุ่มที่คาดหวังลูกค้ายังคงใช้บริการต่อ ถ้าหากหายไปอาจจะส่งผลให้เสียยอดการใช้จ่ายของลูกค้า (รายได้บริษัท) ประมาณ 36% หากต้องเก็บลูกค้าไว้ต่อนั้นย่อมจะต้องทราบถึงข้อมูล Demographic

*Demographic ของลูกค้ากลุ่ม 08.At risk*
![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/23ea43b7-2bda-4822-b9ba-695171b67b3d)



*Demographic ของลูกค้ากลุ่ม 09.Can’t lose them*
![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/4b990003-7330-46e5-ace0-5fe848065406)



*Demographic ของลูกค้ากลุ่ม 10.Hibernating*
![image](https://github.com/AuntiesSnail/DADS5001-Mini-Project/assets/137598346/a5dff853-191e-4784-8a42-8f6a42872bfa)


## **Part 4 : Summary**

* สรุป Demographic ลูกค้าโดยเปรียบเทียบกับข้อมูลการปิดบัญชีของลูกค้า

* สรุป Demographic churning

* Action plan

จาก พฤติกรรม TreeMap พฤติกรรม 8 9 10 ส่วน 1 2 3 เป็นลูกค้าชั้นดี ส่วน 4 5 6 ก็พยายามทำให้มันเป็น 1 2 3 โดยวิเคราะห์จากจุดอ่อนตาม Segment ความหมาย

## **Reference:**

[1] https://how-many-steps-inc.webflow.io/rfm-segmentation-overview

[2] https://www.putler.com/rfm-analysis/

[3] https://www.linkedin.com/pulse/rfm-analysis-important-revenue-growth-analytics-capability-kakas/
