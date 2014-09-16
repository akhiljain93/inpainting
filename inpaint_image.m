clear all;
close all;
clc;
     
delete('input_img.txt');
delete('output_img.txt');
delete('surf_input_img.txt');
delete('surf_output_img.txt');

input_image=imread('disp2_cones.pgm');
input_gray=edge(input_image,'canny');
input_image=double(input_image);
figure,imshow(input_gray,[]);

plane_pts_threshold=50;
thresh=2;
window=15;
threshold=30;

original_img=input_image;
s=size(input_image);
     
[BW,xi,yi] = roipoly(uint8(input_image));
BW1=1-BW;

tic;

[black_row, black_col]=find(BW);
     
for i=1:length(black_row)
    input_image(black_row(i),black_col(i))=0;
    input_gray(black_row(i),black_col(i))=0;
end

selected_area=BW.*255;
size_missing_area=length(find(selected_area));

damged_img=input_image;
     
[row_miss, col_miss]=find(selected_area);

min_row=min(black_row);
min_col=min(black_col);
max_row=max(black_row);
max_col=max(black_col);

chk_mat=input_gray(min_row-window:max_row+window,min_col-window:max_col+window);

if(length(find(chk_mat))>10)
    test_mat=zeros(s(1),s(2));
    img=zeros(s(1),s(2));
    selected_area=zeros(s(1),s(2));
     
    test_mat(min_row-window:max_row+window,min_col-window:max_col+window)=chk_mat;
    selected_area(min_row-window:max_row+window,min_col-window:max_col+window)=1;
    
    figure,imshow(test_mat,[])
    
    edge_cnt=1;
    edge_cnter=0;
     
    seg_rgn=zeros(s(1),s(2),10);
    seg_rgn1=zeros(s(1),s(2),10);
    
    %%%%%%%%%%%TENSOR VOTING BASED REFINING%%%%%%%%%%%%%%%%% 
    sigma=3;
    eta=7;
    T = find_features(test_mat,sigma);
    [e1,e2,l1,l2] = convert_tensor_ev(T);
    cur_sal=l1-l2;
       
    clear r;
    clear c;
       
    [r, c]=find(cur_sal<eta);
    
    for i=1:length(r)
        test_mat(r(i),c(i))=0;
    end
           
    T = convert_tensor_ev(e1,e2,l1,l2);

    test_mat=bwareaopen(test_mat,2);
     
    figure,imshow(test_mat,[])

    while(1)     
        img=zeros(s(1),s(2));    
        final=zeros(s(1),s(2));
        edge_map=bwlabel(test_mat);
        number_of_edges=max(max(edge_map));
        if(isempty(find(test_mat, 1)))
            break;
        end
     
        for ptr=1:1
            [row, col]=find(edge_map==ptr);
            for index=1:length(row)
                img(row(index),col(index))=255;
            end
     
            fact1=var(row);
            fact2=var(col);
            mm=1;

            match_options=[(ptr+1):number_of_edges];
            for ind=(ptr+1):number_of_edges
                [row1, col1]=find(edge_map==ind);
                fact3=var(row1);
                fact4=var(col1);
                for index=1:length(row1)
                    img(row1(index),col1(index))=255;
                end

                [img_row, img_col]=find(img==255);
                
                %%%%%%%%%%%%% ellipse fit %%%%%%%%%%%%%%%%%%%%%%%%%%%
                for index=1:length(img_row)
                    curve_pts(:,index)=[img_row(index);img_col(index)];
                end

                pp1=fact1+fact3;
                pp2=fact2+fact4;

                clear w;
                clear A;
                clear b;

                if(pp1>=pp2)
                    for index=1:length(img_row)
                        A(:,index)=[curve_pts(1,index)^2 curve_pts(1,index) 1];
                        b(index)=[curve_pts(2,index)];
                        w(index)=cur_sal(curve_pts(1,index),curve_pts(2,index));
                    end

                else
                    for index=1:length(img_row)
                        A(:,index)=[curve_pts(2,index)^2 curve_pts(2,index) 1];
                        b(index)=[curve_pts(1,index)];
                        w(index)=cur_sal(curve_pts(1,index),curve_pts(2,index));
                    end
                end

                p=lscov(A',b',w);
                o=1;
                seg=zeros(s(1),s(2));
                seg1=zeros(s(1),s(2));

                if(var(curve_pts(1,:))>=var(curve_pts(2,:)))
                    for i=1:s(1)
                        for j=1:s(2)
                            coef=j-p(1)*i^2-p(2)*i-p(3);
                            if(coef>=0)
                                seg(i,j)=5;
                            else
                                seg(i,j)=10;
                            end
                        end
                    end
                else
                    for i=1:s(1)
                        for j=1:s(2)
                            coef=i-p(1)*j^2-p(2)*j-p(3);
                            if(coef>=0)
                                seg(i,j)=5;
                            else
                                seg(i,j)=10;
                            end
                        end
                    end
                end

                seg1=edge(seg,'canny');

                img1=zeros(s(1),s(2));
                img2=zeros(s(1),s(2));
                img3=zeros(s(1),s(2));
                img4=zeros(s(1),s(2));

                for index=1:length(row)
                    img1(row(index),col(index))=255;
                end
     
                for index=1:length(row1)
                    img2(row1(index),col1(index))=255;
                end
     
                img3=img1.*seg1;
                img4=img2.*seg1;
     
                match_cnt(mm)=length(find(img3))*length(find(img4));
                matched_imgs(:,:,mm)=seg1;

                clear r;
                clear c;

                mm=mm+1;

                for index=1:length(row1)
                    img(row1(index),col1(index))=0;
                end

                o=1;

                clear row1;
                clear col1;
                clear A;
                clear nx;
                clear ny;
                clear img_row;
                clear img_col;
                clear T;
                clear p;
                clear curve_pts;
                clear T;
                clear l1;
                clear l2;
                clear z;
                clear r;
                clear c;
                clear seg1;
            end

            length(find(test_mat))

            if(mm>1)
                min_err=max(match_cnt);
            else
                min_err=threshold+1;
            end

            if(min_err<=threshold | mm==1 | number_of_edges==1)
                [cur_row cur_col]=find(edge_map==ptr);
                rre=zeros(s(1),s(2));

                for tt=1:length(cur_row)
                    rre(cur_row(tt),cur_col(tt))=255;
                end

                rre=bwmorph(rre,'thin','inf');

                [lin_row lin_col]=find(rre);

                curve_pts(:,1)=[lin_row(1);lin_col(1)];
                curve_pts(:,2)=[lin_row(round(length(lin_row)/2));lin_col(round(length(lin_row)/2))];

                slope=(curve_pts(2,2)-curve_pts(2,1))/(curve_pts(1,2)-curve_pts(1,1));
                intercept=curve_pts(2,1)-slope*curve_pts(1,1);

                seg=zeros(s(1),s(2));
                seg1=zeros(s(1),s(2));

                for i=1:s(1)
                    for j=1:s(2)
                        coef=j-slope*i-intercept;
                        if(coef>=0)
                            seg(i,j)=5;
                        else
                            seg(i,j)=10;
                        end
                    end
                end

                seg1=edge(seg,'canny');
                conn_rgn=seg1.*selected_area;
                input_gray=input_gray+conn_rgn;
                figure,imshow(input_gray);
                %%%%%%%%%%%%%%%
                clear curve_pts;

                [con_row con_col]=find(conn_rgn);

                if(length(con_row)>0)
                    edge_cnter=edge_cnter+1;
                    ptr=edge_cnter;

                    for index=1:length(con_row)
                        curve_pts(:,index)=[con_row(index);con_col(index)];
                    end

                    %%%%%%%%%%%%%% polynimial fit %%%%%%%%%%
                    if(var(curve_pts(1,:))>=var(curve_pts(2,:)))
                        p=polyfit(curve_pts(1,:),curve_pts(2,:),2);

                        for i=1:length(row_miss)
                            coef=col_miss(i)-p(1)*row_miss(i)^2-p(2)*row_miss(i)-p(3);
                            if(coef>=0)
                                seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt;
                            else 
                                seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt+1;
                            end
                        end

                        o=1;

                        for i=min_row-window:max_row+window
                            for j=min_col-window:max_col+window
                                coef=j-p(1)*i^2-p(2)*i-p(3);
                                if(coef>=0)
                                    seg_rgn1(i,j,ptr)=edge_cnt;
                                else
                                    seg_rgn1(i,j,ptr)=edge_cnt+1;
                                end
                            end
                        end

                    else
                        p=polyfit(curve_pts(2,:),curve_pts(1,:),2);

                        for i=1:length(row_miss)
                            coef=row_miss(i)-p(1)*col_miss(i)^2-p(2)*col_miss(i)-p(3);
                            if(coef>=0)
                                seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt;
                            else
                                seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt+1;
                            end
                        end

                        o=1;

                        for i=min_row-window:max_row+window
                            for j=min_col-window:max_col+window
                                coef=i-p(1)*j^2-p(2)*j-p(3);
                                if(coef>=0)
                                    seg_rgn1(i,j,ptr)=edge_cnt;
                                else
                                    seg_rgn1(i,j,ptr)=edge_cnt+1;
                                end
                            end
                        end
                    end
                end

                for index=1:length(cur_row)
                    test_mat(cur_row(index),cur_col(index))=0;
                end

                clear lin_row;
                clear lin_col;
                clear cur_row;
                clear cur_col;
                clear con_row;
                clear con_col;
                clear curve_pts;

                break;
            end

            for match_no=1:length(match_cnt)
                if(match_cnt(match_no)==min_err)
                    break;
                end
            end

            seg1=matched_imgs(:,:,match_no);
            conn_rgn=seg1.*selected_area;

            [row col]=find(edge_map==match_options(match_no));

            for index=1:length(row)
                test_mat(row(index),col(index))=0;
            end

            [row col]=find(edge_map==ptr);

            for index=1:length(row)
                test_mat(row(index),col(index))=0;
            end

            matx=seg1.*test_mat;

            clear r;
            clear c;
            [r c]=find(matx);

            for i=1:length(r)
                test_mat(r(i),c(i))=0;
            end

            test_mat=bwareaopen(test_mat,2);
            o=1;

            input_gray=input_gray+conn_rgn;
            figure,imshow(input_gray);

            length(find(test_mat))
            oo=1;

            clear row;
            clear col;
            clear match_cnt;
            clear egde_map;
            clear curve_pts;
            clear matched_imgs;

            [row col]=find(conn_rgn);

            if(length(row)>0)
                edge_cnter=edge_cnter+1;
                ptr=edge_cnter;
                offset=round(length(row)/5);

                for index=1:length(row)
                    curve_pts(:,index)=[row(index);col(index)];
                end
         
                %%%%%%%%%%%%%% polynimial fit %%%%%%%%%%
                if(var(curve_pts(1,:))>=var(curve_pts(2,:)))
                    p=polyfit(curve_pts(1,:),curve_pts(2,:),2);

                    for i=1:length(row_miss)
                        coef=col_miss(i)-p(1)*row_miss(i)^2-p(2)*row_miss(i)-p(3);
                        if(coef>=0)
                            seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt;
                        else
                            seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt+1;
                        end
                    end

                    o=1; 
                    
                    for i=min_row-window:max_row+window
                        for j=min_col-window:max_col+window
                            coef=j-p(1)*i^2-p(2)*i-p(3);
                            if(coef>=0)
                                seg_rgn1(i,j,ptr)=edge_cnt;
                            else
                                seg_rgn1(i,j,ptr)=edge_cnt+1;
                            end
                        end
                    end

                else
                    p=polyfit(curve_pts(2,:),curve_pts(1,:),2);

                    for i=1:length(row_miss)
                        coef=row_miss(i)-p(1)*col_miss(i)^2-p(2)*col_miss(i)-p(3);    
                        if(coef>=0)
                            seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt;
                        else
                            seg_rgn(row_miss(i),col_miss(i),ptr)=edge_cnt+1;
                        end
                    end

                    o=1;

                    for i=min_row-window:max_row+window
                        for j=min_col-window:max_col+window
                            coef=i-p(1)*j^2-p(2)*j-p(3);
                            if(coef>=0)
                                seg_rgn1(i,j,ptr)=edge_cnt;
                            else
                                seg_rgn1(i,j,ptr)=edge_cnt+1;
                            end
                        end
                    end
                end
            end
            %%%%%%%%%%%% End of polynomial fit %%%%%%%%%%%%%%%%%%%%%%
            o=1;
            edge_cnt=edge_cnt+2;

            clear p;
            clear curve_pts;
            clear row;
            clear col;
            clear conn_rgn;
            clear seg1;
        end
    end

    o=1;
         
    seg_rgn_1=zeros(s(1),s(2),edge_cnter);
    seg_rgn_2=zeros(s(1),s(2),edge_cnter);
         
    seg_rgn_1=seg_rgn(:,:,1:edge_cnter);
    seg_rgn_2=seg_rgn1(:,:,1:edge_cnter);
         
    clear seg_rgn;
    clear seg_rgn1;
         
    seg_rgn=seg_rgn_1;
    seg_rgn1=seg_rgn_2;
         
    %%%%%%%%%%%%%%%%% Finding all present segment combinations%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [row_comb, col_comb]=find(seg_rgn(:,:,1));
    index=1;
    for i=1:length(row_comb)
        if(index>1)
            comb_vect=zeros(edge_cnter,1);
            comb_vect(:)=seg_rgn(row_comb(i),col_comb(i),:);

            for j=1:index-1
                comb_match_arr(j)=norm(comb_mat(:,j)-comb_vect);
            end

            if(length(find(comb_match_arr==0))>0)
                continue;
            else
                comb_mat(:,index)=comb_vect;
                index=index+1;
            end
        end
           
        if(index==1)
            comb_mat(:,index)=seg_rgn(row_comb(i),col_comb(i),:);
            index=index+1;
        end
    end
         
    numb_of_comb=size(comb_mat);
    o=1;
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
    clear row_miss;
    clear col_miss;
    [row_miss col_miss]=find(seg_rgn1(:,:,1));
         
    ee1=zeros(s(1),s(2));
    for i=1:length(row_miss)
        ee1(row_miss(i),col_miss(i))=255;
    end

    for cnt=1:numb_of_comb(2)
        img_vect=zeros(1,edge_cnter);
        img_vect(:)=comb_mat(:,cnt);

        depth=0;
        ind=0;
        for ii=1:length(row_miss)
            img_vect1=zeros(1,edge_cnter);
            img_vect1(:)=seg_rgn1(row_miss(ii),col_miss(ii),:);

            if(norm(img_vect-img_vect1)<0.1 & input_image(row_miss(ii),col_miss(ii))>0)
                depth=depth+input_image(row_miss(ii),col_miss(ii));
                ind=ind+1;
            end
        end
         
        avg_depth=depth/ind;
        avg_depth_arr(cnt)=avg_depth;
    end
         
    o=1;

    for cnt=1:numb_of_comb(2)
        img_vect=zeros(1,edge_cnter);
        img_vect1=zeros(1,edge_cnter);
        [row_seg col_seg]=find(seg_rgn(:,:,1));
        mm=max(avg_depth_arr);

        for kk=1:numb_of_comb(2)
            if(avg_depth_arr(kk)==mm)
                break;
            end
        end

        img_vect(:)=comb_mat(:,kk);
        avg_depth_arr(kk)=0;

        img=zeros(s(1),s(2));
        img1=zeros(s(1),s(2));
        img2=zeros(s(1),s(2));
           
        ind=1;
        ind1=1;
        for ii=1:length(row_miss)
            img_vect1(:)=seg_rgn1(row_miss(ii),col_miss(ii),:);
            img1(row_miss(ii),col_miss(ii))=255;

            if(norm(img_vect-img_vect1)<0.1 & input_image(row_miss(ii),col_miss(ii))>0)
                r_rgn(ind)=row_miss(ii);
                c_rgn(ind)=col_miss(ii);
                img(row_miss(ii),col_miss(ii))=255;
                ind=ind+1;
            end
                       
            if(norm(img_vect-img_vect1)<0.1 & input_image(row_miss(ii),col_miss(ii))==0)
                miss_r_rgn(ind1)=row_miss(ii);
                miss_c_rgn(ind1)=col_miss(ii);
                img2(row_miss(ii),col_miss(ii))=255;
                ind1=ind1+1;
            end
        end
         
        o=1;

        if(ind>1)
            Y=zeros(s(1),s(2));
            index=1;
            for k=1:length(r_rgn)
                class(:,index)=[r_rgn(k); c_rgn(k); input_image(r_rgn(k),c_rgn(k))];
                Y(r_rgn(k),c_rgn(k))=255;
                index=index+1;
            end

            size_class=size(class);
            data_dim=size_class(1);
            figure,imshow(Y,[]);

            assign_class=cat(1,class,zeros(1,size_class(2)));
            T=zeros(s(1),s(2));
            ind=1;
            thresh=20;
            cntr=1;
            loop_cnt=1;

            while(1)
                loop_cnt=loop_cnt+1;
                if(loop_cnt>200)
                    break;
                end

                ww=find(assign_class((data_dim+1),:)==0);
                length(ww);

                if(length(ww)<10)
                    break;
                end

                dp_vect=assign_class(1:3,ww(1));
                ptr=1;

                for j=1:size_class(2)
                    eul_dist=norm(dp_vect-assign_class(1:3,j));

                    if(eul_dist<=thresh & assign_class((data_dim+1),j)==0)
                        test_class(:,ptr)=assign_class(1:3,j);
                        arr_ind(ptr)=j;
                        ptr=ptr+1;
                    end
                end
                o=1;

                size_test_class=size(test_class);

                if(size_test_class(2)<50)
                    for j=1:size_test_class(2)
                        Y(test_class(1,j),test_class(2,j))=0;
                        assign_class(4,arr_ind(j))=100;
                    end

                    if(length(find(Y))<plane_pts_threshold)
                        break;
                    end

                    clear test_class;
                    clear ww;
                    clear arr_ind;
                    
                    continue;
                end

                T=zeros(s(1),s(2));
                index=1;
                for j=1:size_test_class(2)
                    T(test_class(1,j),test_class(2,j))=input_image(test_class(1,j),test_class(2,j));
                    index=index+1;
                end

                plane_pts_threshold=50;
                data_dim=size_test_class(1);

                if(size_test_class(2)>plane_pts_threshold)
                    new_class=test_class;
                    new_class=cat(1,test_class,zeros(1,size_test_class(2)));
                    fid = fopen('surf_input_img.txt', 'wt');
                    fprintf(fid, '%d \n',data_dim);
                    fprintf(fid, '%f %f %f %f \n', new_class);
                    fclose(fid);    
         
                    status = dos('NDTensorvoting.exe surf_input_img.txt surf_output_img.txt -scale 80');
                    input_array=textread('surf_output_img.txt','%f');

                    %%%%%%%%%%%%%%%% eigen values %%%%%%%%%%%%%%%%%%%%%%%%%%%
                    eig_val_mat=zeros(size_test_class(1),size_test_class(2));
                    ptr=1;
                    index=2;

                    while(index<=length(input_array))
                        index=index+data_dim+1;
                        k=1;
                        for j=index:(index+data_dim-1)
                            eig_val_mat(k,ptr)=input_array(index);
                            k=k+1;
                            index=index+1;
                        end
                        ptr=ptr+1;
                        index=index+data_dim^2;
                    end

                    tensor_sizes=sum(eig_val_mat(1:data_dim,:));
                    %%%%%%%%%%%%%%%%%%% for getting all the eigen vectors %%%%%%%%%%%
                    index=2;
                    ptr=1;

                    while(index<=length(input_array))
                        index=index+2*data_dim+1;
                        for k=1:data_dim
                            eig_vect_mat(:,ptr)=input_array(index:index+data_dim-1);
                            ptr=ptr+1;
                            index=index+data_dim;
                        end
                    end
                    
                    size_vect=size(eig_vect_mat);
                    %%%%%%%%%%%% for getting eigen vectors with highest eig val %%%%%%
                    index=1;
                    for i=1:data_dim:size_vect(2)
                        highest_eig_vect(:,index)=eig_vect_mat(:,i);
                        index=index+1;
                    end

                    index=1;
                    for i=2:data_dim:size_vect(2)
                        highest_eig_vect1(:,index)=eig_vect_mat(:,i);
                        index=index+1;       
                    end
         
                    index=1;
                    for i=3:data_dim:size_vect(2)
                        highest_eig_vect2(:,index)=eig_vect_mat(:,i);
                        index=index+1;
                    end

                    size_highest_vect=size(highest_eig_vect);  

                    ev=zeros(data_dim,data_dim);
                    for i=1:size_highest_vect(2)
                        vect1=zeros(data_dim,1);
                        vect1(:)=highest_eig_vect(:,i);
                        vect2=zeros(data_dim,1);
                        vect2(:)=highest_eig_vect1(:,i);
                        vect3=zeros(data_dim,1);
                        vect3(:)=highest_eig_vect2(:,i);

                        ev=ev+vect1*vect1';
                    end

                    ev=ev/size_highest_vect(2);
                    [t1 t2]=eigs(ev);
                    eigen_val=[t2(1,1) t2(2,2) t2(3,3)];
                    mm=max(eigen_val);

                    for i=1:length(eigen_val)
                        if(eigen_val(i)==mm)
                            break;
                        end
                    end

                    normal_dir=[t1(1,i); t1(2,i); t1(3,i)];

                    for ii=1:size_highest_vect(2)
                        dot_prod_op(ii)=sum(normal_dir.*highest_eig_vect(:,ii));
                    end
           
                    mm=max(dot_prod_op);
                    for ii=1:size_highest_vect(2)
                        if(dot_prod_op(ii)==mm)
                            break;
                        end
                    end
           
                    plane_pt=[test_class(1,ii);test_class(2,ii);test_class(3,ii)];
                    d=-1*(normal_dir(1)*plane_pt(1)+normal_dir(2)*plane_pt(2)+normal_dir(3)*plane_pt(3));
                    local_plane_info(:,cntr)=[normal_dir(1);normal_dir(2);normal_dir(3);d];

                    for i=1:size_class(2)
                        fact=normal_dir(1)*class(1,i)+normal_dir(2)*class(2,i)+normal_dir(3)*class(3,i)+d;
                        if(abs(fact)<1)
                            assign_class(4,i)=cntr;
                            Y(class(1,i),class(2,i))=0;
                        end
                    end

                    cntr=cntr+1;
                    figure,imshow(Y,[]);

                    if(length(find(Y))<plane_pts_threshold)
                        break;
                    end
           
                    o=1;
           
                    clear new_class;
                    clear eig_vect_mat;
                    clear eig_val_mat;
                    clear highest_eig_vect;
                    clear normal_dir;
                    clear d;
                    clear cov_mat;
                    clear mean_vect;
                    clear l1;
                    clear l2;
                    clear z;
                    clear U;
                    clear p;
                    clear curve_pts;
                    clear test_class;
                    clear arr_ind;
                end
            end

            o=1;
            close all;
         
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ee1=zeros(s(1),s(2));
         
            clear class;
            clear new_class;
            index=1;
            for ii=1:length(row_miss)
                img_vect1(:)=seg_rgn1(row_miss(ii),col_miss(ii),:);
                if(norm(img_vect1-img_vect)<0.1 & input_image(row_miss(ii),col_miss(ii))==0)
                    ee1(row_miss(ii),col_miss(ii))=255;
                    for ff=1:(cntr-1)
                        normal_dir=local_plane_info(1:3,ff);
                        d=local_plane_info(4,ff);

                        if(normal_dir(3)~=0)
                            Z=-1*((normal_dir(1)/normal_dir(3))*row_miss(ii)+(normal_dir(2)/normal_dir(3))*col_miss(ii)+(d/normal_dir(3)));
                        else
                            Z=plane_pt(3);
                        end
                     
                        class(:,index)=[row_miss(ii);col_miss(ii);Z];
                        index=index+1;
                    end
                end
            end

            size_class=size(class);
            class_lim=size_class(2);
            new_ind=1;
            
            for ii=1:length(row_miss)
                img_vect1(:)=seg_rgn1(row_miss(ii),col_miss(ii),:);
                if(norm(img_vect1-img_vect)<0.1 & input_image(row_miss(ii),col_miss(ii))>0)
                    class(:,index)=[row_miss(ii);col_miss(ii);input_image(row_miss(ii),col_miss(ii))];
                    sigma_class(:,new_ind)=[row_miss(ii);col_miss(ii);input_image(row_miss(ii),col_miss(ii))];
                    index=index+1;  
                    new_ind=new_ind+1;
                    ee1(row_miss(ii),col_miss(ii))=255;
                end
            end

            min_pt=[min(sigma_class(1,:)) min(sigma_class(2,:)) min(sigma_class(3,:))];
            max_pt=[max(sigma_class(1,:)) max(sigma_class(2,:)) max(sigma_class(3,:))];
         
            sigma=round(1*norm(min_pt-max_pt))
            new_class=class;
         
            size_class=size(class);
            data_dim=size_class(1);
            new_class=cat(1,class,zeros(1,size_class(2)));
         
            fid = fopen('input_img.txt', 'wt');
            fprintf(fid, '%d \n',data_dim);
            fprintf(fid, '%f %f %f %f \n', new_class);
            fclose(fid);

            status = dos('NDTensorvoting.exe input_img.txt output_img.txt -scale 400');
            input_array=textread('output_img.txt','%f');

            %%%%%%%%%%%%%%%% eigen values %%%%%%%%%%%%%%%%%%%%%%%%%%%55
            eig_val_mat=zeros(size_class(1),size_class(2));
            ptr=1;
            index=2;

            while(index<=length(input_array))
                index=index+data_dim+1;
                k=1;

                for j=index:(index+data_dim-1)
                    eig_val_mat(k,ptr)=input_array(index);
                    k=k+1;
                    index=index+1;
                end

                ptr=ptr+1;
                index=index+data_dim^2;
            end

            surf_sal=eig_val_mat(1,:)-eig_val_mat(2,:);

            for i=1:(cntr-1):class_lim
                arr=surf_sal(i:i+(cntr-1)-1);
                mm=max(arr);

                for ff=1:(cntr-1)
                    if(arr(ff)==mm)
                        break;
                    end
                end
                input_image(class(1,i),class(2,i))=class(3,i+(ff-1));
            end

            figure,imshow(uint8(input_image))
            pause(0.5);

            clear r_rgn;    
            clear c_rgn;  
            clear miss_r_rgn;    
            clear miss_c_rgn;  
            clear highest_eig_vect;  
            clear highest_eig_vect1;
            clear highest_eig_vect2;
            clear input_array;
            clear eig_vect_mat;  
            clear eig_val_mat;  
            clear class;
            clear new_class;
            clear row_seg;
            clear col_seg;
            clear dot_prod_op;
            clear assign_class;
            clear input_array;
            clear local_plane_info;
            clear normal_dir;
            clear d;
           
            rr=ones(s(1),s(2));
            rr=abs(seg_rgn(:,:,1)-(input_image+seg_rgn(:,:,1)));
            [zero_row, zero_col]=find(rr==0);
           
            length(zero_row)
        end
         
        o=1;
         
        if(length(zero_row)==0)
            break;
        end
    end
else
    test_mat=zeros(s(1),s(2));
    window=15;
    plane_pts_threshold=150;

    test_mat(round(min(yi))-window:round(max(yi))+window,round(min(xi))-window:round(max(xi))+window)=255;
    data_mat=input_image(round(min(yi))-window:round(max(yi))+window,round(min(xi))-window:round(max(xi))+window);
    [row_miss col_miss]=find(test_mat);

    ind=1;
    ind1=1;
    for ii=1:length(row_miss)
        img1(row_miss(ii),col_miss(ii))=255;
        if(input_image(row_miss(ii),col_miss(ii))>0)
            r_rgn(ind)=row_miss(ii);
            c_rgn(ind)=col_miss(ii);
            img(row_miss(ii),col_miss(ii))=255;
            ind=ind+1;
        end
    end
       
    Y=zeros(s(1),s(2));    
    index=1;
    for k=1:length(r_rgn)
        class(:,index)=[r_rgn(k); c_rgn(k); input_image(r_rgn(k),c_rgn(k))];
        Y(r_rgn(k),c_rgn(k))=255;
        index=index+1;
    end
    
    size_class=size(class);
    data_dim=size_class(1);
     
    figure,imshow(Y,[]);

    assign_class=cat(1,class,zeros(1,size_class(2)));
    T=zeros(s(1),s(2));
    ind=1;
    thresh=20;
    cntr=1;
    loop_cnt=1;
    while(1)
        loop_cnt=loop_cnt+1;
        if(loop_cnt>50)
            break;
        end
        ww=find(assign_class((data_dim+1),:)==0);
        length(ww);
       
        if(length(ww)<10)
            break;
        end

        dp_vect=assign_class(1:3,ww(1));
        ptr=1;
        
        for j=1:size_class(2)
            eul_dist=norm(dp_vect-assign_class(1:3,j));
            if(eul_dist<=thresh & assign_class((data_dim+1),j)==0)
                test_class(:,ptr)=assign_class(1:3,j);
                arr_ind(ptr)=j;
                ptr=ptr+1;
            end
        end

        o=1;
     
        size_test_class=size(test_class);
        if(size_test_class(2)<50)
            for j=1:size_test_class(2)
                Y(test_class(1,j),test_class(2,j))=0;
                assign_class(4,arr_ind(j))=100;
            end

            if(length(find(Y))<plane_pts_threshold)
                break;
            end
     
            clear test_class;
            clear ww;
            clear arr_ind;
            continue;
        end

        T=zeros(s(1),s(2));
        index=1;
        
        for j=1:size_test_class(2)
            T(test_class(1,j),test_class(2,j))=input_image(test_class(1,j),test_class(2,j));
            index=index+1;
        end
     
        plane_pts_threshold=50;
        data_dim=size_test_class(1);
     
        if(size_test_class(2)>plane_pts_threshold)
            new_class=test_class;
            new_class=cat(1,test_class,zeros(1,size_test_class(2)));

            fid = fopen('surf_input_img.txt', 'wt');
            fprintf(fid, '%d \n',data_dim);
            fprintf(fid, '%f %f %f %f \n', new_class);
            fclose(fid);

            status = dos('NDTensorvoting.exe surf_input_img.txt surf_output_img.txt -scale 80');
            input_array=textread('surf_output_img.txt','%f');

            %%%%%%%%%%%%%%%% eigen values %%%%%%%%%%%%%%%%%%%%%%%%%%%55
            eig_val_mat=zeros(size_test_class(1),size_test_class(2));
            ptr=1;
            index=2;

            while(index<=length(input_array))
                index=index+data_dim+1;
                k=1;

                for j=index:(index+data_dim-1)
                    eig_val_mat(k,ptr)=input_array(index);
                    k=k+1;
                    index=index+1;
                end
        
                ptr=ptr+1;
                index=index+data_dim^2;
            end
     
            tensor_sizes=sum(eig_val_mat(1:data_dim,:));
            index=2;
            ptr=1;

            while(index<=length(input_array))
                index=index+2*data_dim+1;

                for k=1:data_dim
                    eig_vect_mat(:,ptr)=input_array(index:index+data_dim-1);
                    ptr=ptr+1;
                    index=index+data_dim;
                end
            end

            size_vect=size(eig_vect_mat);
     
            %%%%%%%%%%%% for getting eigen vectors with highest eig val %%%%%%
            index=1;
            for i=1:data_dim:size_vect(2)
                highest_eig_vect(:,index)=eig_vect_mat(:,i);
                index=index+1;
            end
     
            index=1;
            for i=2:data_dim:size_vect(2)
                highest_eig_vect1(:,index)=eig_vect_mat(:,i);
                index=index+1;
            end

            index=1;
            for i=3:data_dim:size_vect(2)
                highest_eig_vect2(:,index)=eig_vect_mat(:,i);
                index=index+1;
            end

            size_highest_vect=size(highest_eig_vect);

            ev=zeros(data_dim,data_dim);
            for i=1:size_highest_vect(2)
                vect1=zeros(data_dim,1);
                vect1(:)=highest_eig_vect(:,i);
                vect2=zeros(data_dim,1);
                vect2(:)=highest_eig_vect1(:,i);
                vect3=zeros(data_dim,1);
                vect3(:)=highest_eig_vect2(:,i);
                ev=ev+vect1*vect1';
            end

            ev=ev/size_highest_vect(2);
            [t1 t2]=eigs(ev);
            eigen_val=[t2(1,1) t2(2,2) t2(3,3)];
            mm=max(eigen_val);
       
            for i=1:length(eigen_val)
                if(eigen_val(i)==mm)
                    break;
                end
            end

            normal_dir=[t1(1,i); t1(2,i); t1(3,i)];
            for ii=1:size_highest_vect(2)
                dot_prod_op(ii)=sum(normal_dir.*highest_eig_vect(:,ii));
            end
       
            mm=max(dot_prod_op);
            for ii=1:size_highest_vect(2)
                if(dot_prod_op(ii)==mm)
                    break;
                end
            end
       
            plane_pt=[test_class(1,ii);test_class(2,ii);test_class(3,ii)];
            d=-1*(normal_dir(1)*plane_pt(1)+normal_dir(2)*plane_pt(2)+normal_dir(3)*plane_pt(3));
            local_plane_info(:,cntr)=[normal_dir(1);normal_dir(2);normal_dir(3);d];

            for i=1:size_class(2)
                fact=normal_dir(1)*class(1,i)+normal_dir(2)*class(2,i)+normal_dir(3)*class(3,i)+d;
                if(abs(fact)<1.5)
                    assign_class(4,i)=cntr;
                    Y(class(1,i),class(2,i))=0;
                end
            end
       
            cntr=cntr+1;
            figure,imshow(Y,[]);

            if(length(find(Y))<plane_pts_threshold)
                break;
            end
       
            o=1;
       
            clear new_class;
            clear eig_vect_mat;
            clear eig_val_mat;
            clear highest_eig_vect;
            clear normal_dir;
            clear d;
            clear cov_mat;
            clear mean_vect;
            clear l1;
            clear l2;
            clear z;
            clear U;
            clear p;
            clear curve_pts;
            clear test_class;
            clear arr_ind;
        end
    end
     
    o=1;
     
    close all;
    clear class;
    clear new_class;
    
    index=1;
    for ii=1:length(row_miss)
        if(input_image(row_miss(ii),col_miss(ii))==0)
            for ff=1:(cntr-1)
                normal_dir=local_plane_info(1:3,ff);
                d=local_plane_info(4,ff);
                           
                if(normal_dir(3)~=0)
                    Z=-1*((normal_dir(1)/normal_dir(3))*row_miss(ii)+(normal_dir(2)/normal_dir(3))*col_miss(ii)+(d/normal_dir(3)));
                else
                    Z=plane_pt(3);
                end
                 
                class(:,index)=[row_miss(ii);col_miss(ii);Z];
                index=index+1;
            end
        end
    end  
     
    size_class=size(class);
    class_lim=size_class(2);
     
    new_ind=1;
    for ii=1:length(row_miss)
        if(input_image(row_miss(ii),col_miss(ii))>0)
            class(:,index)=[row_miss(ii);col_miss(ii);input_image(row_miss(ii),col_miss(ii))];
            sigma_class(:,new_ind)=[row_miss(ii);col_miss(ii);input_image(row_miss(ii),col_miss(ii))];
            index=index+1;
            new_ind=new_ind+1;
        end
    end

    min_pt=[min(sigma_class(1,:)) min(sigma_class(2,:)) min(sigma_class(3,:))];
    max_pt=[max(sigma_class(1,:)) max(sigma_class(2,:)) max(sigma_class(3,:))];
    sigma=round(1.8*norm(min_pt-max_pt))
     
    new_class=class;
    size_class=size(class);
    data_dim=size_class(1);
    new_class=cat(1,class,zeros(1,size_class(2)));
     
    fid = fopen('input_img.txt', 'wt');
    fprintf(fid, '%d \n',data_dim);
    fprintf(fid, '%f %f %f %f \n', new_class);
    fclose(fid);

    status = dos('NDTensorvoting.exe input_img.txt output_img.txt -scale sigma');
    input_array=textread('output_img.txt','%f');

    %%%%%%%%%%%%%%%% eigen values %%%%%%%%%%%%%%%%%%%%%%%%%%%55
    eig_val_mat=zeros(size_class(1),size_class(2));
    ptr=1;
    index=2;

    while(index<=length(input_array))
        index=index+data_dim+1;
        k=1;

        for j=index:(index+data_dim-1)
            eig_val_mat(k,ptr)=input_array(index);
            k=k+1;
            index=index+1;
        end

        ptr=ptr+1;
        index=index+data_dim^2;
    end

    surf_sal=eig_val_mat(1,:)-eig_val_mat(2,:);

    for i=1:(cntr-1):class_lim
        arr=surf_sal(i:i+(cntr-1)-1);
        mm=max(arr);

        for ff=1:(cntr-1)
            if(arr(ff)==mm)
                break;
            end
        end  
     
        input_image(class(1,i),class(2,i))=class(3,i+(ff-1));
    end

    clear row;
    clear col;
     
    [row, col]=find(original_img==0);

    for i=1:length(row)
        input_image(row(i),col(i))=0;
    end

    figure,imshow(uint8(input_image))
    
    o=1;
end
toc;