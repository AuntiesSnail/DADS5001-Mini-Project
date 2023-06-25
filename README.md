# **Mini-Project: Credit Card Customer Churning**

## **Part 1 : Introduction**

Customer Churn Rate คืออัตราที่ลูกค้าหยุดใช้บริการในช่วงระยะเวลาใดเวลาหนึ่ง ซึ่งมีผลในการดำเนินธุรกิจที่ต้องการรักษาลูกค้าให้ใช้บริการกับเราให้ยาวนานและมากที่สุด ดังนั้น การป้องกันการยกเลิกใช้บริการของลูกค้าจึงมีความสำคัญมาก โดยการนำ Data ที่เป็นฐานลูกค้า มาจัดเป็น Customer Segment ด้วย RFM Model ร่วมกับข้อมูล Demographic เพื่อทำการวิเคราะห์และวางแผนการทำ Marketing กับลูกค้าในแต่ละกลุ่มให้ได้ตรงตามเป้าหมายและมีประสิทธิภาพมากที่สุด

ซึ่งชุดข้อมูลที่ใช้ในโปรเจคนี้ เป็นข้อมูลลูกค้า Consumer Credit Card Portfolio ของธนาคารแห่งหนึ่ง ที่ต้องการวิเคราะห์หาลูกค้าที่มีแนวโน้มจะยกเลิกการใช้บริการบัตรเครดิต เพื่อที่จะได้มอบการบริการที่ดีขึ้นและกระตุ้นการตัดสินใจของลูกค้าให้กลับมาใช้บริการอีกครั้ง

**ข้อมูล Dataset :**

kaggle : Predicting Credit Card Customer Segmentation

Credit: The original authors – Data Source

License: CC0 1.0 Universal (CC0 1.0) - Public Domain Dedication

No Copyright - You are allowed to copy, modify, distribute, and perform the work, even for commercial purposes, without the need for permission

**Hypothesis :**

การพิจารณา Customer Segment โดย RFM Model ร่วมกับข้อมูล Demographic จะช่วยทำให้เข้าใจพฤติกรรมของลูกค้าได้ดียิ่งขึ้น โดยเฉพาะกลุ่ม “At risk”, “Can’t lose them” และ “Hibernating” ที่มีแนวโน้มจะเลิกใช้บริการ ทำให้สามารถปรับกลยุทธ์ทางการตลาดและเสนอสิทธิพิเศษให้กับลูกค้าในแต่ละกลุ่มได้อย่างเหมาะสม


## **Part 2 : Exploration Data Analysis (EDA)**

**2.1 Review Data:**

ข้อมูลลูกค้าทั้งหมด 10,127 ราย ประกอบด้วย 21 ตัวแปร ได้แก่
Variable	Definition	Variable Group
CLIENTNUM	รหัสลูกค้าธนาคาร	
Attrition_Flag	แสดงสถานะการใช้บริการของลูกค้า	
Customer_Age	อายุลูกค้า	Demographic
Gender	เพศลูกค้า	Demographic
Dependent_count	จำนวนผู้ที่อยู่ในความอุปการะ	
Education_Level	ระดับการศึกษา	Demographic
Marital_Status	สถานภาพสมรส	Demographic
Income_Category	ช่วงรายได้ของลูกค้า	Demographic
Card_Category	ประเภทบัตรของลูกค้า	
Months_on_book	จำนวนเดือนที่ลูกค้าใช้บริการ	
Total_Relationship_Count	จำนวนผลิตภัณฑ์ที่ลูกค้าถือครองกับธนาคาร	Demographic
Months_Inactive_12_mon	จำนวนเดือนที่ลูกค้าไม่มีการเคลื่อนไหว ใน 12 เดือนล่าสุด	
Contacts_Count_12_mon	จำนวนครั้งที่มีการติดต่อ ใน 12 เดือนล่าสุด	
Credit_Limit	วงเงินบัตรเครดิต	
Total_Revolving_Bal	ยอดค้างชำระ	
Avg_Open_To_Buy	วงเงินเครดิตที่พร้อมใช้สำหรับการซื้อสินค้าหรือทำธุรกรรมใหม่ (คำนวณค่าเฉลี่ยของวงเงินเครดิตที่เปิดให้ใช้งานในระยะเวลา 12 เดือน)	
Total_Amt_Chng_Q4_Q1	ยอดใช้จ่าย Q4 เทียบกับ Q1	
Total_Trans_Amt	ยอดใช้จ่ายทั้งหมด ใน 12 เดือนล่าสุด	
Total_Trans_Ct	จำนวนรายการทั้งหมด ใน 12 เดือนล่าสุด	
Total_Ct_Chng_Q4_Q1	จำนวนรายการ Q4 เทียบกับ Q1	
Avg_Utilization_Ratio	ยอดการใช้จ่ายต่อวงเงิน เป็นอัตราเท่าไหร่ 
กำหนดให้ 1 คือใช้จ่ายเต็มวงเงิน และ 0 คือไม่มีการใช้จ่าย คิดเป็นค่าเฉลี่ยจากทั้งหมด 	















2.2 Categorical and Numerical Variable:
ในข้อมูลตัวแปร Attrition_Flag มี Attrited Customer (ลูกค้าที่ยกเลิกบริการ) 1,627 ราย (16.07%) และ Existing Customer (ลูกค้าที่ยังใช้บริการ) 8,500 ราย (83.93%)




























2.3 การวิเคราะห์ RFM Model
	RFM Analysis เป็น Model พื้นฐานเพื่อแบ่งกลุ่มลูกค้าหรือ Customer Segmentation ตามพฤติกรรมการใช้จ่าย โดยตัวแปรที่นำมาวิเคราะห์มี 3 ตัวแปรหลัก คือ R F และ M
	Recency   (R) 	คือระยะเวลาที่ลูกค้าใช้งานบัตร ซึ่งจะใช้ตัวแปร Months_on_book 
	Frequency (F) 	คือความถี่ในการใช้จ่ายของลูกค้า ซึ่งจะใช้ตัวแปร Total_Trans_C
	Monetary  (M) 	คือยอดธุรกรรมทั้งหมด ซึ่งจะใช้ตัวแปร Total_Trans_Amt 

ตารางแสดงข้อมูลตัวอย่าง

