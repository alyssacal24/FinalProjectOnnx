async function runExample() {
    // Create an array to store input values
    var x = [];

    // Retrieve input values from HTML input elements
    x[0] = document.querySelector('.school:checked').value || '';
    x[1] = document.querySelector('.sex:checked').value || '';
    x[2] = document.querySelector('.age').value || '';
    x[3] = document.querySelector('.address:checked').value || '';
    x[4] = document.querySelector('.famsize:checked').value || '';
    x[5] = document.querySelector('.Pstatus:checked').value || '';
    x[6] = document.querySelector('.Medu').value || '';
    x[7] = document.querySelector('.Fedu').value || '';
    x[8] = document.querySelector('.Mjob').value || '';
    x[9] = document.querySelector('.Fjob').value || '';
    x[10] = document.querySelector('.reason').value || '';
    x[11] = document.querySelector('.guardian:checked').value || '';
    x[12] = document.querySelector('.traveltime').value || '';
    x[13] = document.querySelector('.studytime').value || '';
    x[14] = document.querySelector('.failures').value || '';
    x[15] = document.querySelector('.schoolsup:checked').value || '';
    x[16] = document.querySelector('.famsup:checked').value || '';
    x[17] = document.querySelector('.paid:checked').value || '';
    x[18] = document.querySelector('.activities:checked').value || '';
    x[19] = document.querySelector('.nursery:checked').value || '';
    x[20] = document.querySelector('.higher:checked').value || '';
    x[21] = document.querySelector('.internet:checked').value || '';
    x[22] = document.querySelector('.romantic:checked').value || '';
    x[23] = document.querySelector('.famrel').value || '';
    x[24] = document.querySelector('.freetime').value || '';
    x[25] = document.querySelector('.goout').value || '';
    x[26] = document.querySelector('.Dalc').value || '';
    x[27] = document.querySelector('.Walc').value || '';
    x[28] = document.querySelector('.health').value || '';
    x[29] = document.querySelector('.absences').value || '';
    x[30] = document.querySelector('.G1').value || '';
    x[31] = document.querySelector('.G2').value || '';

    // Create a Float32Array from the input values
    let tensorX = new onnx.Tensor(x, 'float32', [1, 32]);

    // Create a new inference session
    let session = new onnx.InferenceSession();

    // Load the ONNX model
    await session.loadModel("./DLnet_StudentPerformance.onnx");

    // Run the model with the input tensor
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output1');

    // Display the output tensor values in the HTML
    let predictions = document.getElementById('predictions');
    predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Predicted Student Performance  </td>
       <td id="td0">  ${outputData.data[0].toFixed(2)}  </td>
     </tr>
  </table>`;
}
