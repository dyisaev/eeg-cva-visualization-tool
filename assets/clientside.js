
window.dash_clientside = Object.assign({}, window.dash_clientside, {

    clientside: {
        figure: function (fig_dict, current_time,threshold,rejections) {
            
            if (!fig_dict) {
                throw "Figure data not loaded, aborting update."
            }
            
            // Copy the fig_data so we can modify it
            fig_dict_copy = {...fig_dict};
            fps=30
            fig_dict_copy["layout"]["shapes"][0]["x0"] = current_time*fps;
            fig_dict_copy["layout"]["shapes"][0]["x1"] = current_time*fps;
            fig_dict_copy["layout"]["uirevision"]=1
            
            const triggered = dash_clientside.callback_context.triggered.map(t => {
                return t.prop_id;
            });
            if ((triggered=='pred-threshold-slider.value') || (triggered=='pred-rejected-events.data')) {
                fig_dict_copy['data'][8]['y']=this.threshold_and_reject(fig_dict_copy['data'][7]['y'],threshold,rejections)
            }
            return fig_dict_copy

        },
        threshold_and_reject: function (predictions,threshold,rejections){
            labels = predictions.map((element,index)=>{
                return element>threshold ? 1 : 0
            })
            rejections.forEach(element => {
                labels.fill(0,element[0],element[1])                
            });
            return labels
        },
        process_rejection: function (pred_reject,pred_reset_rejections,fig_dict,current_time,rejections,fps){
            const triggered = dash_clientside.callback_context.triggered.map(t => {
                return t.prop_id;
            });
            if (triggered=='pred-reset-rejections.n_clicks'){
                return []
            }
            cur_frame_num=Math.round(current_time*fps)
            label_timeseries=fig_dict['data'][8]['y']
            cur_label=label_timeseries[cur_frame_num]
            rejection_start=-1
            rejection_end=-1
            if (cur_label==1) {
                frame_start=cur_frame_num-1
                frame_end=cur_frame_num+1
                for (let index = 0; index < label_timeseries.length; index++) {
                    if (label_timeseries[frame_start]==0) {
                        rejection_start=frame_start
                    }
                    if (label_timeseries[frame_end]==0) {
                        rejection_end=frame_end
                    }
                    if (rejection_start<0) {
                        frame_start=frame_start-1
                    }
                    if (rejection_end<0){
                        frame_end=frame_end+1
                    }
                    if (rejection_start>0 && rejection_end>0) {
                        break
                    }
                }
                rejections.push([rejection_start,rejection_end])            
            }
            return rejections
        
        }
    }
});