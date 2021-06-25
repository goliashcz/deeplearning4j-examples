/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.quickstart.modeling.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

@SuppressWarnings("DuplicatedCode")
public class RoadSegmentClassifierFromSavedModel {

    private static Logger log = LoggerFactory.getLogger(RoadSegmentClassifierFromSavedModel.class);

    public static void main(String[] args) throws  Exception {

        // open the network file
        File locationToSave = new File("/home/goliash/ml_model_91_az");
        MultiLayerNetwork neuralNet = null;
        log.info("Trying to open the model");
        try {
            neuralNet = ModelSerializer.restoreMultiLayerNetwork(locationToSave, true);
            log.info("Success: Model opened");
        } catch (IOException e) {
            throw new Exception(String.format("Unable to open model from %s because of error %s", locationToSave.getAbsolutePath(), e.getMessage()));
        }

        log.info("Loading test data");
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new File("/home/goliash/ml_dist_az_diff.csv")));

        int labelIndex = 14;
        int numClasses = 1;
        int batchSize = 200;

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();

        log.info("Normalizing test data");
        DataNormalization normalizer = ModelSerializer.restoreNormalizerFromFile(locationToSave);
        normalizer.transform(allData);

        log.info("Classifying examples");
        INDArray output = neuralNet.output(allData.getFeatures());
        log.info("Outputing the classifications");
        if(output == null || output.isEmpty())
            throw new Exception("There is no output");
        System.out.println(output);

        Evaluation eval = new Evaluation(1);
        eval.eval(allData.getLabels(), output);
        log.info(eval.stats());
    }
}

