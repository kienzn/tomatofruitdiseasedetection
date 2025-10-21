import cv2
import numpy
from ultralytics import YOLO
from flask import Flask, render_template, request
from PIL import Image
import base64
import os
import io
import time
app = Flask(__name__) # khởi tạo ứng dụng flask, biến name là biến đặc biệt dc hỗ trợ giúp flask xác dịnh xem file nào là file chạy chính
# Khởi tạo model yolo dc train r
model_path = "best9.pt"
#assert os.path.exists(model_path)
#model = YOLO(model_path,task='detect')
# Nhãn bệnh song ngữ
label_vietnam_english= {"Anthracnose":"Bệnh thán thư",
                        "Blossom_End_Rot":"Bệnh thối đáy quả",
                        "Catfaced":" Dị dạng mặt mèo",
                        "Fruit_Cracking": " Vết nứt trên quả",
                        "Healthy_Tomato": "Cà chua khoẻ mạnh",
                        "Late_Blight": "Bệnh cháy muộn",
                        "Mold": "Bệnh mốc",
                        "Spotted_Wilt_Virus":"Bệnh héo đốm"}
# Thông tin chi tiết của từng bệnh
disease_status_info ={
          "Anthracnose":{
              "Triệu_chứng":["Vết bệnh ban đầu là một đốm nhỏ,hơi lõm,ướt bề mặt vỏ quả",
                             "Vết bệnh thường là hình tròn,dạng lõm, phân ranh giới giữa mô bệnh là 1 đường màu đen chạy dọc theo vết bệnh"],
              "Biện_pháp":["Thu gom và tiêu hủy những trái bệnh"]},
          "Blossom_End_Rot":{
              "Triệu_chứng":["Thối đen ở phần đáy đít quả"],
              "Biện_pháp":["Tưới nước đều và tránh úng nước",
                           "Cung cấp đủ canxi cho cây cà chua"]},
          "Catfaced":{
              "Triệu_chứng":[
                  "Quả mắc bệnh này thường có hình khối chia ra nhiều thùy(vấu)",
                  "với các vết sẹo màu nâu xuất hiện giữa các vấu ăn sâu vào trong thịt quả"],
              "Biện_pháp":["Trồng các giống cà chua chịu được nhiệt độ tốt",
                           "Bón phân hợp lý, không dư đạm",
                           "Không nên sử dụng các loại thuốc diệt cỏ,tránh gây tổn thương cơ học cho cây"]},
          "Fruit_Cracking":{
              "Triệu_chứng":["Vết nứt tròn quang quả hoặc kéo dài theo chiều dọc",
                             "Thường xảy ra khi thay đổi độ ẩm đột ngột sau hạn"],
              "Biện_pháp":["Tưới nước đều đặn không để cây thiếu nước"]},
          "Healthy_Tomato":{
              "Triệu_chứng":["Cây không có triệu chứng bệnh"],
              "Biện_pháp" :["Theo dõi, chăm sóc thường xuyên"] },
          "Late_Blight":{
               "Triệu_chứng":["Vết thối màu nâu đen, nhũn nước",
                              "Vết bệnh lan rộng, làm quả rụng sớm"],
               "Biện_pháp":["Không để quả chín lâu trên cây",
                            "Loại bỏ quả khi có triệu chứng bệnh thối nâu nhũn",
                            "Trồng giống cây kháng bệnh, cách nhau một khoảng thích hợp"]},
          "Mold":{
               "Triệu_chứng":["Xuất hiện các lớp mốc trắng mốc xám ở quả và thường trong điều kiện ẩm ướt cao",
                              "Mốc xám là xuất hiện các lớp nấm mốc màu xám như lớp nhung bao phủ lên quả",
                              "Mốc trắng là xuất hiện các vết mốc trắng như bông(sợi nấm)"],
               "Biện_pháp":["Cắt bỏ bộ phận bị bệnh,vệ sinh tay và dụng cụ cắt tỉa",
                            "Tạo điều kiện thông thoáng khí, tránh độ ẩm cao và không trồng cây quá dày"]},
          "Spotted_Wilt_Virus":{
                "Triệu_chứng":["Xuất hiện các đốm vòng màu vàng"],
                "Biện_pháp":["Kiểm soát bọ trĩ"]}}


@app.route('/', methods = ['GET','POST'])   # khi truy cập web flask sẽ gọi hàm def index với 2 chế độ get hoặc post
def index():
  if request.method == 'POST': # Nếu ng dùng bấm nút gửi ảnh
    start_time = time.time()
    upload_image = request.files.get('image')
    captured_data = request.form.get('captured_image')
    if  upload_image and upload_image.filename != '':
      img =Image.open(upload_image).convert('RGB')
    elif captured_data:
    # Trường hợp người dùng chụp ảnh từ webcam (base64)
      captured_data = captured_data.split(",")[1]  # bỏ phần 'data:image/jpeg;base64,'
      img_bytes = base64.b64decode(captured_data)
      img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    else:
      return render_template("index.html",error = 'No image upload: Không có ảnh tải lên hoặc chụp')
    try:
      img_numpy = numpy.array(img) # chuyển ảnh thành mảng numpy để cv2 đọc được
      start_pred = time.time()
      result = model.predict(img_numpy,conf=0.25,iou = 0.65,agnostic_nms = True)[0]
      end_pred = time.time()
      predict_time = end_pred-start_pred
# Vẽ bounding box từ kết quả của YOLO bằng cv2
      boxes = result.boxes.xyxy.cpu().numpy()
      confs = result.boxes.conf.cpu().numpy()
      clss = result.boxes.cls.cpu().numpy()
      info_list = []
      if len(boxes) == 0:
        end_time=time.time()
        processing_time = end_time - start_time
        _,buf = cv2.imencode('.jpg',cv2.cvtColor(img_numpy,cv2.COLOR_RGB2BGR)) # Nén ảnh thành định dạng mong muốn jpg, chuyển kênh màu về BGR 
        imgbase64 = base64.b64encode(buf).decode('utf-8') #Chuyển mảng byte thành cuỗi kí tự base64 ascii, chuyển mảng base64 thành str để dduaw vô template
        return render_template("index.html",resultimg=imgbase64,error='No object detection:Không phát hiện đối tượng nào',info_list = [],processing_time= f"{processing_time:.2f}giây",predict_time=f"{predict_time:.2f}giây")
      else:
       for box,conf,cls in zip(boxes,confs,clss):
         x_min_pred,y_min_pred,x_max_pred,y_max_pred = box
         label_english= result.names[int(cls)]# lấy tên lớp tiếng anh
         label_vietnam= label_vietnam_english.get(label_english,label_english) # nếu lable_en có trong label_vietnamese_enlish sẽ trả về tiếng vt còn kos ẽ trả về tiếng anh
         label_song_ngu = f"{label_vietnam} ({label_english})"
         confidence = conf*100
         text = (f'{label_english} {confidence:.2f}%')
         # Tính kích thước của box
         box_height = y_max_pred - y_min_pred
         box_width = x_max_pred - x_min_pred
         box_size = (box_height+box_width)/2
# phông và độ dày chữ theo kích thước box
         font = max(0.6,min(0.9,box_size/180))
         thickness = max(1,int(font*2))
         text_y= int(y_min_pred-10)
         if text_y <10:
          text_y = int(y_min_pred+20)
         cv2.rectangle(img_numpy,(int(x_min_pred),int(y_min_pred)),(int(x_max_pred),int(y_max_pred)),(255,0,0),2)
         cv2.putText(img_numpy,text,(int(x_min_pred),text_y),cv2.FONT_HERSHEY_SIMPLEX,font,(0,0,255),thickness)
         
         trieu_chung = disease_status_info.get(label_english, {}).get("Triệu_chứng", ["Không rõ"])
         bien_phap = disease_status_info.get(label_english, {}).get("Biện_pháp", ["Không rõ"])

         info_list.append({
              "label": label_song_ngu,
              "confidence": f"{confidence:.2f}%",
              "trieu_chung": trieu_chung,
              "bien_phap": bien_phap
                    })
    #Chuyển ảnh thành chuổi base64 để hiển thị trên web
      _,buf = cv2.imencode('.jpg',cv2.cvtColor(img_numpy,cv2.COLOR_RGB2BGR)) # Nén ảnh thành định dạng mong muốn jpg, chuyển kênh màu về BGR 
      imgbase64 = base64.b64encode(buf).decode('utf-8') #Chuyển mảng byte thành cuỗi kí tự base64 ascii, chuyển mảng base64 thành str để dduaw vô template
      end_time= time.time()
      processing_time= end_time-start_time
    # Trả kết quả về giao diện
      return render_template("index.html",resultimg = imgbase64,info_list=info_list,processing_time=f"{processing_time:.2f}giây",predict_time=f"{predict_time:.2f}giây")
    except Exception as e:
        return render_template("index.html",error = f"Error_Lỗi khi xử lý ảnh: {str(e)}")
  return render_template("index.html")# nếu là get request chỉ hiển thị form upload
if __name__ == "__main__":
  port= int(os.environ.get("PORT",5000)) #Lấy cổng do render cấp
  app.run(debug=True,host="0.0.0.0",port=port)#Cho phép render truy cập



